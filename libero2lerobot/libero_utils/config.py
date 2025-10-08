LIBERO_FEATURES = {
    "observation.images.image": {
        "dtype": "video",
        "shape": (256, 256,3 ),
        "names": ["height", "width", "channel"],
    },
    "observation.images.wrist_image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (15,),
        "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper", "gripper"]},
    },
    "delta_pose_action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
    },
    "delta_joint_action": {
        "dtype": "float32",
        "shape": (8,),
        "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]},
    },
    "abs_pose_action": {
        "dtype": "float32",
        "shape": (10,),
        "names": {"motors": ["x", "y", "z", "ori_0", "ori_1", "ori_2", "ori_3", "ori_4", "ori_5", "gripper"]},
    },
    "abs_joint_action": {
        "dtype": "float32",
        "shape": (8,),
        "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]},
    },
}
