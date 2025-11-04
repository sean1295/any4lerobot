# Configuration for the specific UMI Zarr dataset structure provided.

# Define the LeRobot features based on the `case_open.zarr.zip` structure
UMI_FEATURES = {
    # Image observations
    "observation.images.camera0_global_rgb": {
        "dtype": "video", # This will be automatically changed to "image" if use_videos=False
        "shape": (224, 224, 3), # HWC format expected by LeRobotDataset.add_frame
        "names": ["height", "width", "rgb"],
    },
    "observation.images.camera0_gripper_rgb": {
        "dtype": "video", # This will be automatically changed to "image" if use_videos=False
        "shape": (224, 224, 3), # HWC format
        "names": ["height", "width", "rgb"],
    },
    # Depth observation - Saved as a float32 array
    "observation.depth.camera0_tactile_depth": { # Changed key prefix to observation.depth
        "dtype": "float32", # Changed from "image"
        "shape": (224, 224, 1), # HWC format (channel dimension added)
        "names": ["height", "width", "channel"],
    },
    # Proprioceptive state: 7 robot joints (assuming 7th is gripper)
    "observation.state": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "gripper"]},
    },
    # Action: Read directly from the dataset (shape 6)
    # Assuming dx, dy, dz, drx, dry, drz - adjust names if incorrect
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": {"motors": ["dx", "dy", "dz", "drx", "dry", "drz"]},
    },
}

# Mapping from LeRobot state feature name(s) to UMI Zarr data keys
UMI_STATE_MAP = {
    'observation.state': [
        'robot_joints', # Shape (T, 7)
    ]
}

# UMI Zarr keys for image/depth data to be loaded
UMI_IMAGE_KEYS = ['camera0_global_rgb', 'camera0_gripper_rgb']
UMI_DEPTH_KEYS = ['camera0_tactile_depth'] # Handle depth separately

