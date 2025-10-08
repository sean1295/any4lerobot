from pathlib import Path

import numpy as np
from h5py import File


def load_local_episodes(input_h5: Path):
    with File(input_h5, "r") as f:
        for demo in f["data"].values():
            demo_len = len(demo["obs/agentview_rgb"])
            delta_pose_action = np.array(demo["delta_pose_actions"])
            abs_pose_action = np.array(demo["abs_pose_actions"])
            delta_joint_action = np.array(demo["delta_joint_actions"])
            abs_joint_action = np.array(demo["abs_joint_actions"])

            state = np.concatenate(
                [
                    np.array(demo["obs/ee_states"]),
                    np.array(demo["obs/joint_states"]),
                    np.array(demo["obs/gripper_states"]),
                ],
                axis=1,
            )
            episode = {
                "observation.images.image": np.array(demo["obs/agentview_rgb"]),
                "observation.images.wrist_image": np.array(demo["obs/eye_in_hand_rgb"]),
                "observation.state": np.array(state, dtype=np.float32),
                # "observation.states.ee_state": np.array(demo["obs/ee_states"], dtype=np.float32),
                # "observation.states.joint_state": np.array(demo["obs/joint_states"], dtype=np.float32),
                # "observation.states.gripper_state": np.array(demo["obs/gripper_states"], dtype=np.float32),
                "delta_pose_action": np.array(delta_pose_action, dtype=np.float32),
                "abs_pose_action": np.array(abs_pose_action, dtype=np.float32),
                "delta_joint_action": np.array(delta_joint_action, dtype=np.float32),
                "abs_joint_action": np.array(abs_joint_action, dtype=np.float32),
            }
            yield [{**{k: v[i] for k, v in episode.items()}} for i in range(demo_len)]
