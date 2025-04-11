import torch
import cv2
import numpy as np
import numbers
import collections
import random


class ComposeMix(object):
    r"""Composes several transforms together. It takes a list of
    transformations, where each element odf transform is a list with 2
    elements. First being the transform function itself, second being a string
    indicating whether it's an "img" or "vid" transform
    Args:
        transforms (List[Transform, "<type>"]): list of transforms to compose.
                                                <type> = "img" | "vid"
    Example:
        >>> transforms.ComposeMix([
        [RandomCropVideo(84), "vid"],
        [torchvision.transforms.ToTensor(), "img"],
        [torchvision.transforms.Normalize(
                   mean=[0.485, 0.456, 0.406],  # default values for imagenet
                   std=[0.229, 0.224, 0.225]), "img"]
    ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs, online=False):
        for t in self.transforms:
            if t[1] == "img":
                for idx, img in enumerate(imgs):
                    if online:
                        for i, traj_im in enumerate(img):
                            imgs[idx, i] = t[0](traj_im)
                    else:
                        imgs[idx] = t[0](img)
            elif t[1] == "vid":
                imgs = t[0](imgs)
            else:
                print("Please specify the transform type")
                raise ValueError
        return imgs

class Scale(object):
    r"""Rescale the input image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_LINEAR``
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        assert isinstance(size, int) or (isinstance(
            size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (numpy.array): Image to be scaled.
        Returns:
            numpy.array: Rescaled image.
        """
        if isinstance(self.size, int):
            h, w = img.shape[:2]
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                if ow < w:
                    return cv2.resize(img, (ow, oh), cv2.INTER_AREA)
                else:
                    return cv2.resize(img, (ow, oh))
            else:
                oh = self.size
                ow = int(self.size * w / h)
                if oh < h:
                    return cv2.resize(img, (ow, oh), cv2.INTER_AREA)
                else:
                    return cv2.resize(img, (ow, oh))
        else:
            return cv2.resize(img, tuple(self.size))

