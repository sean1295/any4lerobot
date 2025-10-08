import argparse
import json
import os

import h5py
import numpy as np
import robosuite.utils.transform_utils as T
import tqdm
import gym_pusht
import gymnasium as gym
import zarr
import gdown
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initializes network layers with orthogonal weights."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    """
    Actor-Critic model for continuous action spaces.
    """
    def __init__(self, env):
        super().__init__()
        # Critic network to estimate the value function
        self.critic = nn.Sequential(
            layer_init(nn.Linear(env.observation_space.shape[0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # Actor network to output the mean of the action distribution
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(env.observation_space.shape[0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.action_space.shape[0]), std=0.01),
        )
        # The log standard deviation for the action distribution
        self.actor_logstd = nn.Parameter(torch.zeros(1, env.action_space.shape[0]))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

def main(args):
    print(f"Regenerating PushT dataset from Zarr to HDF5!")

    # Create target directory
    if os.path.isdir(args.target_dir):
        user_input = input(
            f"Target directory already exists at path: {args.target_dir}\nEnter 'y' to overwrite the directory, or anything else to exit: "
        )
        if user_input != "y":
            exit()
    os.makedirs(args.target_dir, exist_ok=True)

    # Prepare JSON file to record metainfo per episode
    metainfo_json_dict = {}
    metainfo_json_out_path = os.path.join(args.target_dir, f"pusht_metainfo.json")
    with open(metainfo_json_out_path, "w") as f:
        # Just test that we can write to this file (we overwrite it later)
        json.dump(metainfo_json_dict, f)

    # Load the Zarr dataset
    if not os.path.isfile(args.raw_data_path):
        id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
        gdown.download(id=id, output=args.raw_data_path, quiet=False)

    dataset_root = zarr.open(args.raw_data_path, 'r')
    actions = dataset_root['data']['action'][:]
    states = dataset_root['data']['state'][:]
    
    # Get episode start/end indices
    ep_ends_zarr = dataset_root['meta/episode_ends'][:]
    ep_starts = np.concatenate(([0], ep_ends_zarr[:-1]))
    num_episodes = len(ep_ends_zarr)

    # Setup
    num_replays = 0

    # Create new HDF5 file for regenerated demos
    new_data_path = os.path.join(args.target_dir, f"pusht_demo.hdf5")
    new_data_file = h5py.File(new_data_path, "w")
    grp = new_data_file.create_group("data")

    # Initialize environment
    env = gym.make("gym_pusht/PushT-v0",
    obs_type="pixels_agent_pos",
    observation_width=args.resolution,
    observation_height=args.resolution,
    max_episode_steps=400
    )
    env.reset()

    dummy_env = gym.make("gym_pusht/PushT-v0",
    obs_type="state",
    observation_width=args.resolution,
    observation_height=args.resolution,
    max_episode_steps=400
    )
    # model = PPO("MlpPolicy", dummy_env, verbose=0)
    # model.load("/home/dcs3zc/pusht/ppo_pusht_overfit")

    model = Agent(dummy_env).to('cuda')
    model.load_state_dict(torch.load("/home/dcs3zc/pusht/ppo_pusht.pth"))

    avg_final_cov = []

    pbar = tqdm.tqdm(range(num_episodes))

    for i in pbar:
        ep_start_idx = ep_starts[i]
        ep_end_idx = ep_ends_zarr[i]
        
        ep_actions = actions[ep_start_idx:ep_end_idx]
        ep_states = states[ep_start_idx:ep_end_idx]
        
        # Reset environment to the initial state of the episode        
        env.reset(options={"reset_to_state": ep_states[0]})
        env.reset(seed = i)
        obs = env.get_obs()
        reward = 0
        done = 0
        info = None
        
        # Set up new data lists
        images = []
        states_replayed = []
        actions_replayed = []
        rewards = []
        dones = []
        
        # Replay the episode
        for ep_timestep, action in enumerate(ep_actions):   
            if ep_timestep == 0:
                action[:2] = np.array(env.agent.position)
            # Record data
            prev_coverage = info["coverage"] if info is not None else 0.0    
            images.append(obs["pixels"])
            states_replayed.append(obs["agent_pos"])
            actions_replayed.append(action)
            rewards.append(reward)
            dones.append(done)

            # Step environment with the action
            obs, reward, done, _, info = env.step(action)
        
        prev_coverage = info["coverage"]

        # Temporary storage for the 20 steps
        temp_images = []
        temp_states = []
        temp_actions = []
        temp_rewards = []
        temp_dones = []
        

        # Pre-run additional steps
        num_attempts = 500
        add_timesteps = 20

        if add_timesteps > 0:
            for attempt_num in range(num_attempts): 
                future_covs = []
                if attempt_num > 0:
                    env.reset(options={"reset_to_state": ep_states[0]})
                    env.reset(seed = i)     
                    for ep_timestep, action in enumerate(ep_actions):   
                        env.step(action)

                for _ in range(add_timesteps):
                    agent_position = np.array(env.agent.position)
                    block_position = np.array(env.block.position)
                    block_angle = env.block.angle % (2 * np.pi)
                    model_input = np.concatenate([agent_position, block_position, [block_angle]], dtype=np.float64)
                    
                    # Predict action
                    # raw_action, _ = model.predict(model_input, deterministic=False)
                    raw_action, _, _, _ = model.get_action_and_value(torch.from_numpy(model_input).float().to('cuda').unsqueeze(0))
                    raw_action = raw_action[0].cpu().numpy()
                    raw_action += np.random.normal(loc=0, scale=0.01 * (attempt_num), size=raw_action.shape)
                    pos_agent = np.array(env.agent.position)
                    # action = pos_agent + (raw_action-0.5).clip(-0.5, 0.5) * 6
                    action = pos_agent + raw_action * 2
                    
                    # Step environment
                    obs, reward, done, _, info = env.step(action)
                    
                    # Record data from the *current* step
                    temp_images.append(obs["pixels"])
                    temp_states.append(obs["agent_pos"])
                    temp_actions.append(action)
                    temp_rewards.append(reward)
                    temp_dones.append(done)
                    
                    # Append the coverage value from the *current* step
                    future_covs.append(info["coverage"])

                best_timestep_to_keep = -1
                max_valid_coverage = -1

                # Iterate through the recorded data
                for add_timestep in range(add_timesteps):
                    current_coverage = future_covs[add_timestep]
                    time_passed = add_timestep + 1  # Timestep starts from 1
                    
                    # Check if the coverage meets the conditions
                    if current_coverage > prev_coverage + 2e-4 * time_passed:
                        # If it passes, check if this is the best one we've found so far
                        if current_coverage > max_valid_coverage:
                            max_valid_coverage = current_coverage
                            best_timestep_to_keep = time_passed

                final_cov = max(prev_coverage, max_valid_coverage) 
                if final_cov < 0.90:
                    print(f"Trial {attempt_num}:Final coverage is less than 0.90. Best value: {final_cov}")
                    continue
                elif best_timestep_to_keep > 0:      
                    if best_timestep_to_keep > 0:
                        print(f"With additional actions, the best valid coverage is {max_valid_coverage:.4f} at timestep {best_timestep_to_keep}, improving from {prev_coverage:.4f}.")
                    else:
                        print("Storing only original data.")         
                    # Append data up to and including this timestep
                    images.extend(temp_images[:best_timestep_to_keep])
                    states_replayed.extend(temp_states[:best_timestep_to_keep])
                    actions_replayed.extend(temp_actions[:best_timestep_to_keep])
                    rewards.extend(temp_rewards[:best_timestep_to_keep])
                    dones.extend(temp_dones[:best_timestep_to_keep])
                    break
        else:
            final_cov = prev_coverage

        if final_cov < 0.90:
            continue
        
        avg_final_cov.append(final_cov)
        
        # After the episode, save replayed trajectories to new HDF5 files
        ep_data_grp = grp.create_group(f"demo_{i}")
        obs_grp = ep_data_grp.create_group("obs")
        
        # Save observations
        obs_grp.create_dataset("images", data=np.stack(images, axis=0))
        obs_grp.create_dataset("states", data=np.stack(states_replayed, axis=0))
        
        # Save actions, rewards, and dones
        ep_data_grp.create_dataset("actions", data=np.stack(actions_replayed, axis=0))
        ep_data_grp.create_dataset("rewards", data=np.array(rewards, dtype=np.uint8))
        ep_data_grp.create_dataset("dones", data=np.array(dones, dtype=np.uint8))

        # Record metainfo for the episode
        num_replays += 1
        episode_key = f"demo_{i}"
        metainfo_json_dict[episode_key] = {}
        metainfo_json_dict[episode_key]["success"] = True
        metainfo_json_dict[episode_key]["initial_state"] = ep_states[0].tolist()

        # Write metainfo to JSON file
        with open(metainfo_json_out_path, "w") as f:
            json.dump(metainfo_json_dict, f, indent=2)

        pbar.set_description(f"Total # episodes succeeded/replayed: {num_replays}/{i+1}")

    # Close HDF5 file
    new_data_file.close()
    print(f"Dataset regeneration complete! Saved new dataset at: {args.target_dir}")
    print(f"Saved metainfo JSON at: {metainfo_json_out_path}")
    print(f"Final statistics: Average coverage: {np.mean(avg_final_cov):.4f}, Total # episodes succeeded/replayed: {num_replays}/{i+1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=256, help="Resolution of the images. Example: 256")
    parser.add_argument(
        "--raw_data_path",
        type=str,
        help="Path to the raw Zarr dataset file. Example: ./pusht_cchi_v7_replay.zarr",
        required=True,
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        help="Path to regenerated dataset directory. Example: ./pusht_regenerated_hdf5",
        required=True,
    )
    args = parser.parse_args()
    main(args)