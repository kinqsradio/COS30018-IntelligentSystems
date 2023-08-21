import os
import numpy as np
from PIL import Image
import imageio

def dataset(directory, img_width):
    """
    Load the dataset
    Args:
      - directory: path to the data directory
      - img_width: target image width
    Returns:
      - features, labels, tags
    """
    tags = {}
    features = []
    label_list = []

    for tag in os.listdir(directory):
        if tag not in tags:
            tags[tag] = len(tags)
        dir_path = os.path.join(directory, tag)
        for filename in os.listdir(dir_path):
            filepath = os.path.join(dir_path, filename)
            image = imageio.imread(filepath)[..., :3]
            image = np.array(Image.fromarray(image).resize((img_width, img_width)))
            features.append(image)
            label_list.append(tags[tag])

    features = np.array(features)
    label_list = np.array(label_list)

    return features, label_list, list(tags.keys())