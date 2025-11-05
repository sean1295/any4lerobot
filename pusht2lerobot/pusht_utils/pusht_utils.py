from pathlib import Path

import numpy as np
from h5py import File
import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


def load_local_episodes(input_h5: Path):
    with File(input_h5, "r") as f:
        for demo in f["data"].values():
            demo_len = len(demo["obs/images"])
            images = demo["obs/images"]
            states = demo["obs/states"]
            actions = demo["actions"]
            rewards = demo["rewards"]
            dones = demo["dones"]
            print(len(rewards), len(dones), len(states))
            episode = {
                "observation.image": np.array(images),
                "observation.state": np.array(states, dtype=np.float32),
                "action": np.array(actions, dtype=np.float32),
                # "next.done": np.array(dones),
                # "next.reward": np.array(rewards, dtype=np.float32),
            }
            yield [{**{k: v[i] for k, v in episode.items()}} for i in range(demo_len)]

def validate_all_metadata(all_metadata: list[LeRobotDatasetMetadata]):
    """
    implemented by @Cadene
    """
    # validate same fps, robot_type, features

    fps = all_metadata[0].fps
    robot_type = all_metadata[0].robot_type
    features = all_metadata[0].features

    for meta in tqdm.tqdm(all_metadata, desc="Validate all meta data"):
        if fps != meta.fps:
            raise ValueError(f"Same fps is expected, but got fps={meta.fps} instead of {fps}.")
        if robot_type != meta.robot_type:
            raise ValueError(f"Same robot_type is expected, but got robot_type={meta.robot_type} instead of {robot_type}.")
        if features != meta.features:
            raise ValueError(f"Same features is expected, but got features={meta.features} instead of {features}.")

    return fps, robot_type, features
