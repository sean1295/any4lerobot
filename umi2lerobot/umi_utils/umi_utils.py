import zarr
import numpy as np
import logging
from pathlib import Path
# Removed calculate_actions function as actions are loaded directly

def load_umi_episodes(zarr_path: Path, config: dict, state_map: dict, image_keys: list, depth_keys: list):
    """
    Loads data from a UMI Zarr dataset and yields episodes frame by frame.
    Reads actions directly from the 'action' key.
    Loads depth data as float32 array.

    Args:
        zarr_path (Path): Path to the Zarr dataset (.zarr or .zarr.zip).
        config (dict): LeRobot features configuration dictionary.
        state_map (dict): Maps LeRobot state feature name to list of UMI keys.
        image_keys (list): List of UMI keys corresponding to RGB image data.
        depth_keys (list): List of UMI keys corresponding to depth data.


    Yields:
        tuple: (episode_index, frame_data_dict)
               frame_data_dict contains data for a single timestep, formatted for
               lerobot_dataset.add_frame(), including 'observation.state',
               'action', and depth as 'observation.depth.<key>'.
    """
    try:
        store = zarr.ZipStore(str(zarr_path), mode='r') if str(zarr_path).endswith('.zip') else zarr.DirectoryStore(str(zarr_path))
        root = zarr.open(store, mode='r')
        logging.info(f"Successfully opened Zarr dataset at {zarr_path}")
    except Exception as e:
        logging.error(f"Failed to open Zarr dataset at {zarr_path}: {e}")
        return

    data = root['data']
    meta = root['meta']

    episode_ends = meta['episode_ends'][:]
    start_idx = 0

    for ep_idx, end_idx in enumerate(episode_ends):
        logging.info(f"Processing Episode {ep_idx} (frames {start_idx} to {end_idx})...")
        num_frames = end_idx - start_idx
        if num_frames <= 0:
            logging.warning(f"Episode {ep_idx} has {num_frames} frames. Skipping.")
            start_idx = end_idx
            continue

        # --- Load State Data for the Episode ---
        episode_states_parts = []
        state_feature_name = next(iter(state_map)) # Assumes only one combined state feature for now
        states_ok = True
        for umi_key in state_map[state_feature_name]:
            try:
                state_data = data[umi_key][start_idx:end_idx]
                if state_data.shape[0] != num_frames:
                    logging.warning(f"Inconsistent frame count in episode {ep_idx} for state key {umi_key}. Expected {num_frames}, got {state_data.shape[0]}. Skipping episode.")
                    states_ok = False
                    break
                episode_states_parts.append(state_data)
            except KeyError:
                logging.error(f"State key '{umi_key}' not found in Zarr data group for episode {ep_idx}. Skipping episode.")
                states_ok = False
                break
            except Exception as e:
                logging.error(f"Error reading state key '{umi_key}' for episode {ep_idx}: {e}. Skipping episode.")
                states_ok = False
                break

        if not states_ok or not episode_states_parts:
            start_idx = end_idx
            continue # Skip to next episode

        # Concatenate state parts into the full state vector for the episode
        try:
            # Ensure consistent float32 type
            episode_full_states = np.concatenate(episode_states_parts, axis=1).astype(np.float32)
        except ValueError as e:
            logging.error(f"Error concatenating state parts for episode {ep_idx}: {e}. Shapes were: {[p.shape for p in episode_states_parts]}. Skipping episode.")
            start_idx = end_idx
            continue

        # --- Load Action Data for the Episode ---
        try:
            episode_actions = data['action'][start_idx:end_idx].astype(np.float32)
            if episode_actions.shape[0] != num_frames:
                logging.warning(f"Inconsistent frame count for 'action' in episode {ep_idx}. Expected {num_frames}, got {episode_actions.shape[0]}. Skipping episode.")
                start_idx = end_idx
                continue
        except KeyError:
            logging.error(f"Action key 'action' not found in Zarr data group for episode {ep_idx}. Skipping episode.")
            start_idx = end_idx
            continue
        except Exception as e:
            logging.error(f"Error reading 'action' key for episode {ep_idx}: {e}. Skipping episode.")
            start_idx = end_idx
            continue

        # --- Load Image Data for the Episode ---
        episode_images = {}
        images_ok = True
        for umi_img_key in image_keys:
            lerobot_img_key = f"observation.images.{umi_img_key}"
            try:
                img_data = data[umi_img_key][start_idx:end_idx]
                if img_data.shape[0] != num_frames:
                    logging.warning(f"Inconsistent frame count for image key {umi_img_key} in episode {ep_idx}. Expected {num_frames}, got {img_data.shape[0]}. Skipping episode.")
                    images_ok = False
                    break
                if img_data.ndim != 4 or img_data.shape[-1] not in [1, 3, 4]: # T, H, W, C
                    logging.warning(f"Unexpected image shape for {umi_img_key} in episode {ep_idx}: {img_data.shape}. Expected (T, H, W, C). Skipping episode.")
                    images_ok = False
                    break
                episode_images[lerobot_img_key] = img_data.astype(np.uint8) # Ensure uint8
            except KeyError:
                logging.error(f"Image key '{umi_img_key}' not found in Zarr data group for episode {ep_idx}. Skipping episode.")
                images_ok = False
                break
            except Exception as e:
                logging.error(f"Error reading image key '{umi_img_key}' for episode {ep_idx}: {e}. Skipping episode.")
                images_ok = False
                break

        if not images_ok:
            start_idx = end_idx
            continue # Skip to next episode

        # --- Load Depth Data for the Episode ---
        episode_depths = {}
        depths_ok = True
        for umi_depth_key in depth_keys:
            # Use observation.depth.<key> convention
            lerobot_depth_key = f"observation.depth.{umi_depth_key}"
            try:
                depth_data = data[umi_depth_key][start_idx:end_idx]
                if depth_data.shape[0] != num_frames:
                    logging.warning(f"Inconsistent frame count for depth key {umi_depth_key} in episode {ep_idx}. Expected {num_frames}, got {depth_data.shape[0]}. Skipping episode.")
                    depths_ok = False
                    break
                if depth_data.ndim != 3: # Expecting T, H, W
                     logging.warning(f"Unexpected depth shape for {umi_depth_key} in episode {ep_idx}: {depth_data.shape}. Expected (T, H, W). Skipping episode.")
                     depths_ok = False
                     break
                # Add channel dimension and ensure float32
                episode_depths[lerobot_depth_key] = depth_data[..., np.newaxis].astype(np.float32) # Add channel dim -> T, H, W, 1

            except KeyError:
                logging.error(f"Depth key '{umi_depth_key}' not found in Zarr data group for episode {ep_idx}. Skipping episode.")
                depths_ok = False
                break
            except Exception as e:
                logging.error(f"Error reading depth key '{umi_depth_key}' for episode {ep_idx}: {e}. Skipping episode.")
                depths_ok = False
                break

        if not depths_ok:
            start_idx = end_idx
            continue # Skip to next episode


        # --- Yield Frames ---
        for i in range(num_frames):
            frame_data = {
                state_feature_name: episode_full_states[i],
                "action": episode_actions[i],
                **{img_key: episode_images[img_key][i] for img_key in episode_images}, # HWC format
                **{depth_key: episode_depths[depth_key][i] for depth_key in episode_depths} # HWC format (float32)
            }
            yield ep_idx, frame_data

        start_idx = end_idx # Move to the start of the next episode

    # Close the store if it's a ZipStore
    if isinstance(store, zarr.ZipStore):
        store.close()

