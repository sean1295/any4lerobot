PUSHT_FEATURES = {
    "observation.images.image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (2,),
        "names": {"motors": ["x", "y"]},
    },
    "action": {
        "dtype": "float32",
        "shape": (2,),
        "names": {"motors": ["x", "y"]},
    },
}
