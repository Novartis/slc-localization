from PIL import Image

from torch.utils.data import Dataset
import numpy as np


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
        try:
            img = np.asarray(Image.open(img_name))
            image = np.zeros((img.shape[0], img.shape[1], 3), dtype="float")
            image[:, :, 0] = img.copy()
            image[:, :, 1] = img.copy()
            image[:, :, 2] = img.copy()
            image = self.transform(image)

        except Exception:
            print("Error {e}: image file ignored at %s" % img_name)

        return image

    def __len__(self):
        return len(self.img_list)
