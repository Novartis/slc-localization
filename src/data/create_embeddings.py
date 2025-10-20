# Import packages
import os

import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms

import logging

import src.models.models as models
import src.data.transform as transform

from src.data.datasets import slc_dataset
from src.data.utilities import compute_features, process_images


# Get logger (configured in main.py)
logger = logging.getLogger(__name__)


def create_embeddings(data_dir=None):
    """
    Create embeddings for the images in the dataset.
    This function processes images, applies transformations, and computes features using a pre-trained model.
    Args:
        data_dir (str, optional): Base data directory. If None, uses current working directory.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../data')
    else:
        data_dir = os.path.abspath(data_dir)

    # Do we have a GPU available?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"GPU Device: {os.getenv('CUDA_VISIBLE_DEVICES')}")

    # Initialize the model
    model = models.DenseNet()
    model = model.to(device)

    # Use more then 1 GPU (if available)
    if torch.cuda.device_count() > 1:
        logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # Define project directory for images
    root_directory = os.path.join(data_dir, "resolute_imaging/")

    # Define transformation
    tra = [
        transform.resize((1024, 1024)),
        transforms.ToTensor(),
        transform.to_float(),
        transform.normalize(),
    ]

    img_list = process_images(root_directory)

    dataset = slc_dataset(img_list, transform=transforms.Compose(tra))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)

    df_files = pd.DataFrame(img_list, columns=["image_path"])
    df_files["image_name"] = df_files["image_path"].apply(
        lambda x: os.path.basename(x)
    )
    logger.info(f"Files shape: {df_files.shape}")
    features = compute_features(dataloader, model, device)
    df_features = pd.DataFrame(features)
    logger.info(f"Features shape: {df_features.shape}")
    return df_features, df_files
