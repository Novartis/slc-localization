import os
import random
import numpy as np

import torch
from PIL import Image
import glob
import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """
    Args:
        seed (int): seed to set for pseudo-randomization
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed + 100)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed + 50)
    random.seed(seed - 100)
    os.environ["PYTHONHASHSEED"] = str(seed - 50)


def compute_features(dataloader, model, device):
    """
    Compute features from defined model

    Args:
        dataloader: load the data
        model: torch model
        device: GPU or CPU
    """

    model.eval()

    # discard the label information in the dataloader
    feature_list = []
    broken_list = []
    with torch.no_grad():
        for i, input_tensor in enumerate(dataloader):
            if input_tensor is None:  # Skip None tensors
                logger.warning(f"Skipping None tensor at index {i}")
                broken_list.append(i)
                continue
            input_tensor = input_tensor.to(device)
            aux = model(input_tensor).data.cpu().numpy()
            feature_list.append(aux)

            if (i % 10) == 0:
                logger.info("{0} / {1}".format(i, len(dataloader)))

    features = np.concatenate(feature_list, axis=0)
    return features


def process_images(root_dir):
    """
    Loops through all image files ('.tif') within each subfolder of the given root directory.

    Args:
        root_dir (str): The path to the root directory containing the subfolders with images.
    Returns:
        list: List of image file names (not full paths)
    """
    img_list = []
    subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    for subfolder in subfolders:
        image_files = glob.glob(
            os.path.join(subfolder, "*.tif")
        )  # Find all .tif files in the subfolder

        for image_file in image_files:
            try:
                img = Image.open(image_file)
                img.close()  # Close the image file to free resources
                img_list.append(image_file)  # Add only the file name if it can be opened
            except Exception:
                logger.warning(f"rm {image_file}")
                continue
            # Add your image processing code here.  For example:
            # from PIL import Image
            # img = Image.open(image_file)
            # # Do something with the image
            # img.close()
    return img_list
