from PIL import Image, UnidentifiedImageError

from torch.utils.data import Dataset
import numpy as np
import logging

# Get logger (configured in main.py)
logger = logging.getLogger(__name__)


class slc_dataset(Dataset):
    def __init__(self, img_list, transform=None):
        """
        Initialize the dataset.

        Args:
            img_list (list): List of image file paths.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        super(slc_dataset, self).__init__()

        self.img_list = img_list
        self.transform = transform

    def __getitem__(self, index):
        img_name = self.img_list[index]
        image = None
        try:
            img = np.asarray(Image.open(img_name))
            image = np.zeros((img.shape[0], img.shape[1], 3), dtype="float")
            image[:, :, 0] = img.copy()
            image[:, :, 1] = img.copy()
            image[:, :, 2] = img.copy()
            image = self.transform(image)

        except (IOError, UnidentifiedImageError, OSError) as e:
            logger.error(f"Error loading image {img_name}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error processing image {img_name}: {e}")

        return image

    def __len__(self):
        return len(self.img_list)
