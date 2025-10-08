from pathlib import Path

import numpy as np
from h5py import File


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
