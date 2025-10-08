"""
Adapted from https://github.com/openvla/openvla/blob/main/experiments/robot/libero/regenerate_libero_dataset.py

Regenerates a LIBERO dataset (HDF5 files) by replaying demonstrations in the environments.

Notes:
    - We save image observations at 256x256px resolution (instead of 128x128).
    - We filter out transitions with "no-op" (zero) actions that do not change the robot's state.
    - We filter out unsuccessful demonstrations.
    - In the LIBERO HDF5 data -> RLDS data conversion (not shown here), we rotate the images by
    180 degrees because we observe that the environments return images that are upside down
    on our platform.

Usage:
    python experiments/robot/libero/regenerate_libero_dataset.py \
        --libero_task_suite [ libero_spatial | libero_object | libero_goal | libero_10 ] \
        --libero_raw_data_dir <PATH TO RAW HDF5 DATASET DIR> \
        --libero_target_dir <PATH TO TARGET DIR>

    Example (LIBERO-Spatial):
        python experiments/robot/libero/regenerate_libero_dataset.py \
            --libero_task_suite libero_spatial \
            --libero_raw_data_dir ./LIBERO/libero/datasets/libero_spatial \
            --libero_target_dir ./LIBERO/libero/datasets/libero_spatial_no_noops

"""

import argparse
import json
import sys
import os
libero_path = '/sfs/gpfs/tardis/home/dcs3zc/LIBERO'
sys.path.append(libero_path)

import h5py
import numpy as np
import robosuite.utils.transform_utils as T
import tqdm
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row.
    
    Args:
        matrix: A NumPy array of rotation matrices with shape (*, 3, 3), where
                * denotes any number of leading batch dimensions.

    Returns:
        A NumPy array of the 6D rotation representation, with shape (*, 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    # Slice the input matrix to get the first two rows (and all columns)
    # The `...` handles any leading batch dimensions.
    sliced_matrix = matrix[..., :2, :]

    # Reshape the sliced matrix to flatten the last two dimensions (2, 3) into 6.
    # The shape is determined by the original batch dimensions plus the new final dimension of 6.
    batch_dim = matrix.shape[:-2]
    new_shape = batch_dim + (6,)
    
    # Use .copy() to ensure the function returns a new array, not a view of the original.
    return sliced_matrix.reshape(new_shape).copy()

def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file, 
        "camera_heights": resolution, 
        "camera_widths": resolution,
        "render_visual_mesh": False
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


def main(args):
    print(f"Regenerating {args.libero_task_suite} dataset!")

    # Create target directory
    if os.path.isdir(args.libero_target_dir):
        user_input = input(
            f"Target directory already exists at path: {args.libero_target_dir}\nEnter 'y' to overwrite the directory, or anything else to exit: "
        )
        if user_input != "y":
            exit()
    os.makedirs(args.libero_target_dir, exist_ok=True)

    # Prepare JSON file to record success/false and initial states per episode
    metainfo_json_dict = {}
    metainfo_json_out_path = os.path.join(args.libero_target_dir, f"{args.libero_task_suite}_metainfo.json")
    with open(metainfo_json_out_path, "w") as f:
        # Just test that we can write to this file (we overwrite it later)
        json.dump(metainfo_json_dict, f)

    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    # Setup
    num_replays = 0
    num_success = 0
    num_noops = 0

    use_noops = args.use_noops

    # controller constants (hard-coded)
    kd = 14.14213562
    kp = 50.0

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task in suite
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, "llava", resolution=args.resolution)

        # Get dataset for task
        orig_data_path = os.path.join(args.libero_raw_data_dir, f"{task.name}_demo.hdf5")
        assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]

        # Create new HDF5 file for regenerated demos
        new_data_path = os.path.join(args.libero_target_dir, f"{task.name}_demo.hdf5")
        new_data_file = h5py.File(new_data_path, "w")
        grp = new_data_file.create_group("data")

        for i in range(len(orig_data.keys())):
            # Get demo data
            demo_data = orig_data[f"demo_{i}"]
            orig_actions = demo_data["actions"][()]
            orig_states = demo_data["states"][()]
            rewards = demo_data["rewards"][()]
            dones = demo_data["rewards"][()]

            # Reset environment, set initial state, and wait a few steps for environment to settle
            env.reset()     
            obs = env.set_init_state(orig_states[0])       

            if use_noops:                
                for _ in range(10):
                    obs, reward, done, info = env.step(get_libero_dummy_action("llava"))

            # Set up new data lists
            states = []
            delta_pose_actions = []
            delta_joint_actions = []
            abs_pose_actions = []
            abs_joint_actions = []
            ee_states = []
            gripper_states = []
            joint_states = []
            robot_states = []
            agentview_images = []
            eye_in_hand_images = []

            # Replay original demo actions in environment and record observations
            for ep_timestep, action in enumerate(orig_actions):
                # Skip transitions with no-op actions
                prev_action = delta_pose_actions[-1] if len(delta_pose_actions) > 0 else None
                if use_noops and is_noop(action, prev_action):
                    print(f"\tSkipping no-op action: {action}")
                    num_noops += 1
                    continue

                if states == []:
                    # In the first timestep, since we're using the original initial state to initialize the environment,
                    # copy the initial state (first state in episode) over from the original HDF5 to the new one
                    states.append(orig_states[0])
                    robot_states.append(demo_data["robot_states"][0])
                else:
                    # For all other timesteps, get state from environment and record it
                    if use_noops:
                        states.append(env.sim.get_state().flatten())
                    else:
                        states.append(orig_states[ep_timestep])
                        obs = env.regenerate_obs_from_state(orig_states[ep_timestep])
                    robot_states.append(
                        np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                    )

                # Record absolute actions (from demo)
                env.env.robots[0].control(action, policy_step=True)
                controller = env.env.robots[0].controller    

                torques = controller.torques
                desired_torque = np.linalg.solve(controller.mass_matrix, torques - controller.torque_compensation)
                joint_pos = np.array(controller.sim.data.qpos[controller.qpos_index])
                joint_vel = np.array(controller.sim.data.qvel[controller.qvel_index])
                position_error = (desired_torque + joint_vel * kd) / kp

                desired_qpos = position_error + joint_pos            
                goal_pos =controller.goal_pos
                goal_ori_6d = matrix_to_rotation_6d(controller.goal_ori)
                gripper_action = (-action[-1:] + 1) / 2 # (-1: open, 1: close) -> (0: close, 1: open)

                abs_pose_action = np.concatenate((goal_pos, goal_ori_6d, gripper_action))
                abs_joint_action = np.concatenate((desired_qpos, gripper_action))
                abs_pose_actions.append(abs_pose_action)
                abs_joint_actions.append(abs_joint_action)

                # Record delta action (from demo)
                delta_pose_action = np.concatenate((action[:-1], gripper_action))
                delta_joint_action = abs_joint_action.copy()
                delta_joint_action[:-1] -= obs["robot0_joint_pos"]
                delta_pose_actions.append(delta_pose_action)
                delta_joint_actions.append(delta_joint_action)
                
                # Record data returned by environment
                if "robot0_gripper_qpos" in obs:
                    gripper_states.append(obs["robot0_gripper_qpos"])
                joint_states.append(obs["robot0_joint_pos"])
                ee_states.append(
                    np.hstack(
                        (
                            obs["robot0_eef_pos"],
                            T.quat2axisangle(obs["robot0_eef_quat"]),
                        )
                    )
                )
                agentview_images.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                eye_in_hand_images.append(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))

                # Execute demo action in environment
                if use_noops:
                    obs, reward, done, info = env.step(action.tolist())         

            if not use_noops or done:
                # At end of episode, save replayed trajectories to new HDF5 files (only keep successes)
                if use_noops:
                    dones = np.zeros(len(delta_pose_actions)).astype(np.uint8)
                    dones[-1] = 1
                    rewards = np.zeros(len(delta_pose_actions)).astype(np.uint8)
                    rewards[-1] = 1
                    assert len(delta_pose_actions) == len(agentview_images)    
                            

                ep_data_grp = grp.create_group(f"demo_{i}")
                obs_grp = ep_data_grp.create_group("obs")
                obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
                obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
                obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
                obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
                obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])
                obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
                obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))
                ep_data_grp.create_dataset("delta_pose_actions", data=delta_pose_actions)
                ep_data_grp.create_dataset("delta_joint_actions", data=delta_joint_actions)
                ep_data_grp.create_dataset("abs_pose_actions", data=abs_pose_actions)
                ep_data_grp.create_dataset("abs_joint_actions", data=abs_joint_actions)
                ep_data_grp.create_dataset("states", data=np.stack(states))
                ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
                ep_data_grp.create_dataset("rewards", data=rewards)
                ep_data_grp.create_dataset("dones", data=dones)
                num_success += 1

            # Record success/false and initial environment state in metainfo dict
            num_replays += 1  
            task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{i}"
            if task_key not in metainfo_json_dict:
                metainfo_json_dict[task_key] = {}
            if episode_key not in metainfo_json_dict[task_key]:
                metainfo_json_dict[task_key][episode_key] = {}
            metainfo_json_dict[task_key][episode_key]["success"] = bool(done) if use_noops else True
            metainfo_json_dict[task_key][episode_key]["initial_state"] = orig_states[0].tolist()

            # Write metainfo dict to JSON file
            # (We repeatedly overwrite, rather than doing this once at the end, just in case the script crashes midway)
            with open(metainfo_json_out_path, "w") as f:
                json.dump(metainfo_json_dict, f, indent=2)

            # Count total number of successful replays so far
            print(
                f"Total # episodes replayed: {num_replays}, Total # successes: {num_success} ({num_success / num_replays * 100:.1f} %)"
            )

            # Report total number of no-op actions filtered out so far
            print(f"  Total # no-op actions filtered out: {num_noops}")
            

        # Close HDF5 files
        orig_data_file.close()
        if len(new_data_file["data"]) == 0:
            new_data_file.close()
            os.remove(new_data_path)
        else:
            new_data_file.close()
        print(f"Saved regenerated demos for task '{task_description}' at: {new_data_path}")

    print(f"Dataset regeneration complete! Saved new dataset at: {args.libero_target_dir}")
    print(f"Saved metainfo JSON at: {metainfo_json_out_path}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=256, help="Resolution of the images. Example: 256")
    parser.add_argument(
        "--libero_task_suite",
        type=str,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO task suite. Example: libero_spatial",
        required=True,
    )
    parser.add_argument(
        "--libero_raw_data_dir",
        type=str,
        help="Path to directory containing raw HDF5 dataset. Example: ./LIBERO/libero/datasets/libero_spatial",
        required=True,
    )
    parser.add_argument(
        "--libero_target_dir",
        type=str,
        help="Path to regenerated dataset directory. Example: ./LIBERO/libero/datasets/libero_spatial_no_noops",
        required=True,
    )
    parser.add_argument(
        "--use_noops",
        action="store_true",
        help="Whether or not to filter no ops (idle actions)",
    )
    args = parser.parse_args()

    # Start data regeneration
    main(args)
