import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, anno_class_img):
        for tf in self.transforms:
            img, anno_class_img = tf(img, anno_class_img)
        return img, anno_class_img


"""
Handle Scale image
"""


class Scale(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img, anno_class_img):
        print("Scaled")
        # Get width and Height of image
        width = img.size[0]
        height = img.size[1]

        # random scale value from first elememt to second element
        scale = np.random.uniform(self.scale[0], self.scale[1])

        # Resize width and height of original image with scale
        scale_width = int(width * scale)
        scale_height = int(height * scale)

        # Resize annotation image with scale
        img = img.resize((scale_width, scale_height), Image.BICUBIC)
        anno_class_img = anno_class_img.resize((scale_width, scale_height), Image.NEAREST)

        # Processing image while scale
        if scale > 1.0:
            # Define left location
            left = scale_width - width
            left = int(np.random.uniform(0, left))

            # Define top location
            top = scale_height - height
            top = int(np.random.uniform(0, top))

            # Crop image
            img = img.crop((left, top, left + width, top + height))
            anno_class_img = anno_class_img.crop((left, top, left + width, top + height))

        else:
            # Make a copy of the image
            p_pallette = anno_class_img.copy().getpalette()  # get pallette pixel
            img_original = img.copy()
            anno_class_img_original = anno_class_img.copy()

            # Define left location
            pad_width = width - scale_width
            pad_width_left = int(np.random.uniform(0, pad_width))

            # Define top location
            pad_height = height - scale_height
            pad_height_top = int(np.random.uniform(0, pad_height))

            # Create new image
            img = Image.new(img.mode, (width, height), (0, 0, 0))  # Create empty image
            # Paste img original to empty image
            img.paste(img_original, (pad_width_left, pad_height_top))

            # Similar create new img
            anno_class_img = Image.new(anno_class_img.mode, (width, height), (0))
            anno_class_img.paste(anno_class_img_original, (pad_width_left, pad_height_top))
            anno_class_img.putpalette(p_pallette)

        return img, anno_class_img


"""
Handle Rotate image
"""


class RandomRotation(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, anno_class_img):
        # Random rotation angle value
        rotate_angle = np.random.uniform(self.angle[0], self.angle[1])

        # Rotate image
        img = img.rotate(rotate_angle, Image.BILINEAR)
        anno_class_img = anno_class_img.rotate(rotate_angle, Image.NEAREST)

        return img, anno_class_img


"""
Handle Mirror image
"""


class RandomMirror(object):
    def __call__(self, img, anno_class_img):
        # Random to decide mirror image or not
        if np.random.randint(2):
            img = ImageOps.mirror(img)
            anno_class_img = ImageOps.mirror(anno_class_img)

        return img, anno_class_img


"""
Handle resize image
"""


class Resize(object):
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, img, anno_class_img):
        # Resize image into round
        img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        anno_class_img = anno_class_img.resize((self.input_size, self.input_size), Image.NEAREST)

        return img, anno_class_img


"""
Handle normalize tensor image
"""


class Normalize_Tensor(object):
    def __init__(self, color_mean, color_std):
        self.color_mean = color_mean
        self.color_std = color_std

    def __call__(self, img, anno_class_img):
        # Transform image to tensor
        img = transforms.functional.to_tensor(img)
        # Normalize tensor
        img = transforms.functional.normalize(img, self.color_mean, self.color_std)

        ### Transform white border (ambiguous) to black border ###
        # Transform annotation image to numpy (array)
        anno_class_img = np.array(anno_class_img)
        # Get the index where it is marked as ambiguous (white line = 255)
        index = np.where(anno_class_img == 255)
        anno_class_img[index] = 0

        # Transform a array numpy to torch
        anno_class_img = torch.from_numpy(anno_class_img)

        return img, anno_class_img
