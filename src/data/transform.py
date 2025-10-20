# Import packages
import torch
import skimage.transform
import numbers


# Convert image to float type
class to_float(object):
    def __call__(self, img):
        img = img.type(torch.FloatTensor)
        return img


# Resize images
class resize(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size), 3)
        else:
            self.size = size

    def __call__(self, img):
        img = skimage.transform.resize(
            img, self.size, mode="reflect", anti_aliasing=True
        )
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(size={})".format(self.size)


# Max scaling
class scale(object):
    def __call__(self, img):
        img = img / img.max()
        return img


# Normalize: subtract mean and dived by std
class normalize(object):
    def __call__(self, img):
        mean = img.mean()
        std = img.std()
        img = (img - mean) / std
        return img
