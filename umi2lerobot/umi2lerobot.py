import argparse
import os
import shutil
from pathlib import Path

import zarr
import numpy as np
import pandas as pd
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import write_info, write_tasks, write_stats
from datatrove.utils.logging import logger


UMI_FEATURES = {
    "observation.images.image": {
        "dtype": "video", "shape": (224, 224, 3), "names": ["height", "width", "channel"], "desc": "global RGB camera view"
    },
    "observation.images.wrist_image": {
        "dtype": "video", "shape": (224, 224, 3), "names": ["height", "width", "channel"], "desc": "gripper RGB camera view"
    },
    # "observation.states.tactile_depth": {
    #     "dtype": "float32", "shape": (224 * 224,), "names": [str(digit) for digit in np.arange(224 * 224)], "desc": "tactile depth map"
    # },
    "observation.state": {
        "dtype": "float32", "shape": (7,), "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]}, "desc": "robot joint positions"
    },
    "action": {
        "dtype": "float32", "shape": (6,), "names": {"motors": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3"]}, "desc": "target action"
    },
}

import numpy as np

def compute_full_stats(episodes):
    """Compute detailed stats (min, max, mean, std, quantiles) per feature."""
    stats = {}

    def flatten_concat(arr_list):
        """Flatten (N, H, W, C) or (N, D) arrays and concatenate along axis 0."""
        return np.concatenate(
            [a.reshape(len(a), -1) for a in arr_list if len(a) > 0], axis=0
        )

    def feature_stats(data):
        """Compute per-dimension summary statistics for a 2D array [N, D]."""
        return {
            "min": np.min(data, axis=0).tolist(),
            "max": np.max(data, axis=0).tolist(),
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(),
            "count": [int(len(data))],
            "q01": np.quantile(data, 0.01, axis=0).tolist(),
            "q10": np.quantile(data, 0.10, axis=0).tolist(),
            "q50": np.quantile(data, 0.50, axis=0).tolist(),
            "q90": np.quantile(data, 0.90, axis=0).tolist(),
            "q99": np.quantile(data, 0.99, axis=0).tolist(),
        }

    # --- Numeric modalities ---
    all_actions = flatten_concat([ep["action"] for ep in episodes])
    all_joints = flatten_concat([ep["robot_joints"] for ep in episodes])
    # all_tactile = flatten_concat([ep["camera0_tactile_depth"] for ep in episodes])

    stats["action"] = feature_stats(all_actions)
    stats["observation.states.robot_joints"] = feature_stats(all_joints)
    # stats["observation.states.tactile_depth"] = feature_stats(all_tactile)

    # --- Frame index and global index stats ---
    frame_indices = []
    global_indices = []
    idx_offset = 0
    for ep in episodes:
        n = len(ep["action"])
        frame_indices.extend(list(range(n)))
        global_indices.extend(list(range(idx_offset, idx_offset + n)))
        idx_offset += n

    stats["frame_index"] = feature_stats(np.array(frame_indices).reshape(-1, 1))
    stats["index"] = feature_stats(np.array(global_indices).reshape(-1, 1))

    # --- Video modalities ---
    def video_feature_stats(video_list):
        samples = []
        for vid in video_list:
            v = vid.astype(np.float32) / 255.0
            if len(v) > 50:  # sample up to 50 frames per episode
                idx = np.linspace(0, len(v) - 1, 50).astype(int)
                v = v[idx]
            samples.append(v.reshape(-1, 3))
        all_pixels = np.concatenate(samples, axis=0)
        return {
            "min": np.min(all_pixels, axis=0).reshape(3, 1, 1).tolist(),
            "max": np.max(all_pixels, axis=0).reshape(3, 1, 1).tolist(),
            "mean": np.mean(all_pixels, axis=0).reshape(3, 1, 1).tolist(),
            "std": np.std(all_pixels, axis=0).reshape(3, 1, 1).tolist(),
            "count": [int(len(all_pixels))],
            "q01": np.quantile(all_pixels, 0.01, axis=0).reshape(3, 1, 1).tolist(),
            "q10": np.quantile(all_pixels, 0.10, axis=0).reshape(3, 1, 1).tolist(),
            "q50": np.quantile(all_pixels, 0.50, axis=0).reshape(3, 1, 1).tolist(),
            "q90": np.quantile(all_pixels, 0.90, axis=0).reshape(3, 1, 1).tolist(),
            "q99": np.quantile(all_pixels, 0.99, axis=0).reshape(3, 1, 1).tolist(),
        }

    stats["observation.images.image"] = video_feature_stats(
        [ep["camera0_global_rgb"] for ep in episodes]
    )
    stats["observation.images.wrist_image"] = video_feature_stats(
        [ep["camera0_gripper_rgb"] for ep in episodes]
    )

    return stats


def load_umi_zarr(zarr_path: Path):
    """Load a zipped or unzipped UMI-style Zarr dataset."""
    store = zarr.ZipStore(zarr_path, mode='r')
    root = zarr.open(store, mode='r')
    data = root["data"]
    meta = root["meta"]

    episode_ends = meta["episode_ends"][:].tolist()
    start_idx = 0
    episodes = []

    for end_idx in episode_ends[:]:
        episode = {
            "action": data["action"][start_idx:end_idx],
            "camera0_global_rgb": data["camera0_global_rgb"][start_idx:end_idx],
            "camera0_gripper_rgb": data["camera0_gripper_rgb"][start_idx:end_idx],
            # "camera0_tactile_depth": data["camera0_tactile_depth"][start_idx:end_idx],
            "robot_joints": data["robot_joints"][start_idx:end_idx],
        }
        episodes.append(episode)
        start_idx = end_idx
    return episodes


def umi_to_lerobot(input_zarr: Path, output_path: Path, task_name: str):
    """Convert one UMI .zarr dataset to LeRobot format with metadata."""
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create dataset and metadata
    dataset = LeRobotDataset.create(
        repo_id=f"{input_zarr.parent.name}/{input_zarr.stem}",
        root=output_path,
        fps=20,
        robot_type="franka",
        features=UMI_FEATURES,
    )
    metadata = dataset.meta

    logger.info(f"Processing {input_zarr}")
    episodes = load_umi_zarr(input_zarr)

    total_frames = 0
    for ep_idx, ep in enumerate(tqdm(episodes, desc="Saving episodes")):
        for i in range(len(ep["action"])):
            frame_data = {
                "observation.images.image": ep["camera0_global_rgb"][i],
                "observation.images.wrist_image": ep["camera0_gripper_rgb"][i],
                # "observation.states.tactile_depth": ep["camera0_tactile_depth"][i].flatten(),
                "observation.state": ep["robot_joints"][i],
                "action": ep["action"][i],
                "task": task_name,
            }        
            dataset.add_frame(frame_data)
            total_frames += 1

        dataset.save_episode()
        logger.info(f"Saved episode {ep_idx+1}/{len(episodes)} "
                    f"({len(ep['action'])} frames)")

    # --- Populate and write metadata ---
    metadata.tasks = pd.DataFrame(
        {"task_index": [0]}, index=[task_name]
    )
    metadata.info.update({
        "total_tasks": 1,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "splits": {"train": f"0:{len(episodes)}"},
    })
    metadata.stats = compute_full_stats(episodes)

    write_tasks(metadata.tasks, metadata.root)
    write_info(metadata.info, metadata.root)
    write_stats(metadata.stats, metadata.root)

    logger.info(f"Finished writing dataset and metadata: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--task-name", type=str, default="umi-task")
    args = parser.parse_args()

    for src in args.src_paths:
        out_dir = args.output_path / src.stem
        umi_to_lerobot(src, out_dir, args.task_name)


if __name__ == "__main__":
    main()
