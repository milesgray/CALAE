import pathlib
import numbers
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
import torchvision.transforms.functional as TF
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def _is_numpy(img):
    return isinstance(img, np.ndarray)

def _is_numpy_image(img):
    return img.ndim in {2, 3}

####################################################################################################################
########### F R A C T A L #############-------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# Custom Fractal Images dataset with high resolution images.  Must apply crop to use.
# ------------------------------------------------------------------------------------------------------------------
class Fractal(Dataset):
    def __init__(self, path="/content/all/", part="train"):
        self.all_data = all_paths = [
            str(p.absolute()) for p in pathlib.Path(path).glob("*")
        ]
        self.total = len(self.all_data)
        if part == "train":
            self.data = self.all_data[: int(self.total * 0.9)]
        else:
            self.data = self.all_data[int(self.total * 0.9) :]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.data[idx]).convert("RGB"))


# ------------------------------------------------------------------------------------------------------------------
# Prepares a set of transformations that crops a certain scale square area randomly from each images
# in a batch, effectively making a much larger dataset than individual image count suggests.
# ------------------------------------------------------------------------------------------------------------------
def make_fractal_alae_dataloader(
    dataset,
    batch_size,
    image_size=4,
    crop_size=512,
    num_workers=3,
    use_grayscale=False,
    crop_mode="random",
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
):
    transform_list = []
    if isinstance(crop_mode, str):
        if crop_mode == "random":
            transform_list.append(
                transforms.RandomCrop(
                    crop_size, pad_if_needed=True, padding_mode="symmetric"
                )
            )
        elif crop_mode == "center":
            transform_list.append(transforms.CenterCrop(crop_size))
    transform_list.append(transforms.Resize((image_size, image_size)))
    transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    transform_list.append(transforms.RandomVerticalFlip(p=0.5))
    transform_list.append(
        transforms.ColorJitter(brightness=0.1, contrast=0.3, saturation=0.3, hue=0.2)
    )
    if use_grayscale:
        transform_list.append(transforms.RandomGrayscale(p=0.1))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean, std, inplace=True))

    dataset.transform = transforms.Compose(transform_list)
    return DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
    )


# ------------------------------------------------------------------------------------------------------------------
# Custom Fractal Images dataset with high resolution images.  Must apply crop and also return the coordinates of
# of the crop in the form of the upper left and lower right points (bounding box - [x1,y1,x2,y2]). Supports the concept
# of multiple crops from each image so that Contrastive Learning can be used with each crop from the same image has a label
# applied in this class based on the index
# ------------------------------------------------------------------------------------------------------------------
class FractalLabel(Dataset):
    def __init__(self, path="/content/all/", part="train"):
        self.all_data = all_paths = [
            str(p.absolute()) for p in pathlib.Path(path).glob("*")
        ]
        self.total = len(self.all_data)
        if part == "train":
            self.data = self.all_data[: int(self.total * 0.9)]
        else:
            self.data = self.all_data[int(self.total * 0.9) :]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        result, coords = self.transform(Image.open(self.data[idx]).convert("RGB"))
        label = torch.full((result.shape[0],), fill_value=idx, dtype=torch.int)
        return (result, label, coords)


# ------------------------------------------------------------------------------------------------------------------
# Prepares a set of transformations that makes many crops of a certain scale square area randomly from each image
# in a batch, effectively making a much larger dataset than individual image count suggests. Also returns the coordinates
# of each crop. Results in a 4-d tensor [N, C, H, W] with N being number of crops
# ------------------------------------------------------------------------------------------------------------------
def make_fractal_clr_dataloader(
    dataset,
    batch_size,
    image_size=4,
    crop_size=512,
    num_workers=3,
    crop_mode=5,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
):
    transform_list = []
    transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    transform_list.append(transforms.RandomVerticalFlip(p=0.5))
    transform_list.append(
        transforms.ColorJitter(brightness=0.1, contrast=0.3, saturation=0.3, hue=0.2)
    )
    # transform_list.append(transforms.RandomGrayscale(p=0.1))
    transform_list.append(MultiCropV2(crop_size, image_size, count=crop_mode))
    transform_list.append(BuildOutput(mean, std))

    dataset.transform = transforms.Compose(transform_list)
    return DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
    )


# ------------------------------------------------------------------------------------------------------------------
# Custom Fractal Images dataset with high resolution images.  Must apply crop and also return the coordinates of
# of the crop in the form of the upper left and lower right points (bounding box - [x1,y1,x2,y2]). Supports the concept
# of multiple crops from each image so that Contrastive Learning can be used with each crop from the same image has a label
# applied in this class based on the index
# ------------------------------------------------------------------------------------------------------------------
class FractalLabelSR(Dataset):
    def __init__(self, path="/content/all/", part="train"):
        self.all_data = all_paths = [
            str(p.absolute()) for p in pathlib.Path(path).glob("*")
        ]
        self.total = len(self.all_data)
        if part == "train":
            self.data = self.all_data[: int(self.total * 0.9)]
        else:
            self.data = self.all_data[int(self.total * 0.9) :]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        result, original, coords = self.transform(
            Image.open(self.data[idx]).convert("RGB")
        )
        label = torch.full((result.shape[0],), fill_value=idx, dtype=torch.int)
        return (result, original, label, coords)


# ------------------------------------------------------------------------------------------------------------------
# Prepares a set of transformations that makes many crops of a certain scale square area randomly from each image
# in a batch, effectively making a much larger dataset than individual image count suggests. Also returns the coordinates
# of each crop. Results in a 4-d tensor [N, C, H, W] with N being number of crops
# ------------------------------------------------------------------------------------------------------------------
def make_fractal_clr_sr_dataloader(
    dataset,
    batch_size,
    image_size=4,
    crop_size=512,
    num_workers=3,
    crop_mode=5,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
):
    transform_list = []
    transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    transform_list.append(transforms.RandomVerticalFlip(p=0.5))
    transform_list.append(
        transforms.ColorJitter(brightness=0.1, contrast=0.3, saturation=0.3, hue=0.2)
    )    
    transform_list.append(MultiCropV2(crop_size, image_size, count=crop_mode, return_original=True))
    transform_list.append(BuildOutput(mean, std, super_res=True))

    dataset.transform = transforms.Compose(transform_list)
    return DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
    )


def _get_image_size(img):
    if _is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


class MultiCrop(object):
    def __init__(
        self,
        crop_size,
        resize_size,
        counts=5,
        padding=None,
        pad_if_needed=False,
        fill=0,
        padding_mode="constant",
        interpolation=Image.BILINEAR,
    ):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size
        self.count = count
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

        if isinstance(resize_size, numbers.Number):
            self.resize_size = (int(resize_size), int(resize_size))
        else:
            self.resize_size = resize_size
        self.interp = interpolation
        self.resizecrop = transforms.Resize(self.resize_size, interpolation=self.interp)

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        img = self._check_size(img)
        results = []
        coords = []
        for i in range(self.count):
            data, coord = self._random_crop(img)
            data = self.resizecrop(data)
            results.append(data)
            coords.append(self._resize_coord(coord))

        return (results, coords)

    def _check_size(self, x):
        """Ensures the image is big enough to"""
        self.h, self.w = _get_image_size(x)
        # if not using padding boundary for valid crop area, then total size is just crop size
        # if use pad is enforced, there is an extra amount of padding that is not valid, so the resulting image is larger
        total_h = self.crop_size[0]
        total_w = self.crop_size[1]
        if self.h < total_h or self.w < total_w:
            pad_amount = 0
            # calculate image size ratio to preserve preportions after resize
            if self.h < self.w:
                # smaller side will be equal to crop size + pad amount
                ratio_h = 1
                # larger side will be scaled up so that it stays larger
                ratio_w = self.w / self.h
                # unified ratio to increase size by based on smaller side
                ratio_r = total_w / self.h
            else:
                ratio_h = self.h / self.w
                ratio_w = 1
                ratio_r = total_h / self.w
            # do resize based on if either PIL or Tensor
            if _is_pil_image(x):
                x = x.resize(
                    int(int(self.w * ratio_r) + pad_amount * ratio_w),
                    int(int(self.h * ratio_r) + pad_amount * ratio_h),
                )
                # get new size
                self.h, self.w = _get_image_size(x)
                return x
            elif isinstance(img, torch.Tensor) and img.dim() > 2:
                x = x.resize(
                    int(int(self.w * ratio_r) + pad_amount * ratio_w),
                    int(int(self.h * ratio_r) + pad_amount * ratio_h),
                )
                # get new size
                self.h, self.w = _get_image_size(x)
                return x
            else:
                # Numpy? shouldn't happen...
                return x
        else:
            # image is large enough already
            return x

    def _random_crop(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.crop_size[1]:
            img = F.pad(
                img, (self.crop_size[1] - img.size[0], 0), self.fill, self.padding_mode
            )
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.crop_size[0]:
            img = F.pad(
                img, (0, self.crop_size[0] - img.size[1]), self.fill, self.padding_mode
            )

        i, j, h, w = self.get_params(img, self.crop_size)

        x1 = i
        y1 = j
        x2 = x1 + h
        y2 = y1 + w

        return TF.crop(img, i, j, h, w), (x1, y1, x2, y2, h, w)

    def _resize_coord(self, coord):
        """Scale the coordinates by the amount the crop was resized"""
        ratio_x = self.resize_size[0] / self.crop_size[1]
        ratio_y = self.resize_size[0] / self.crop_size[1]

        x1 = int(coord[0] * ratio_x)
        y1 = int(coord[1] * ratio_y)
        x2 = int(coord[2] * ratio_x)
        y2 = int(coord[3] * ratio_y)
        h = int(coord[4] * ratio_x)
        w = int(coord[5] * ratio_y)

        return (x1, y1, x2, y2, h, w)


class BuildOutput:
    def __init__(self, mean, std, super_res=False):
        self.mean = mean
        self.std = std
        self.super_res = super_res

    def __call__(self, x):        
        if self.super_res:
            y = x[2] # coords of crop resized imgs
            z = x[1] # original cropped imgs
            x = x[0] # cropped resized imgs            
        else:
            y = x[1] # coords of crop resized imgs
            x = x[0] # cropped resized imgs
        data = torch.stack(
            [
                transforms.Normalize(self.mean, self.std, inplace=True)(
                    torch.from_numpy(
                        np.array(crop, np.float32, copy=False).transpose((2, 0, 1))
                    ).contiguous()
                )
                for crop in x
            ]
        )
        label = torch.Tensor(y)
        if self.super_res:
            original_data = torch.stack(
                [
                    transforms.Normalize(self.mean, self.std, inplace=True)(
                        torch.from_numpy(
                            np.array(crop, np.float32, copy=False).transpose((2, 0, 1))
                        ).contiguous()
                    )
                    for crop in z
                ]
            )

            return data, original_data, label
        else:
            return data, label

class MultiCropV2(object):   
    def __init__(self, crop_size, resize_size, 
                 count=5,
                 padding=None, 
                 return_original=False,
                 pad_if_needed=False, 
                 fill=0, 
                 padding_mode='constant', 
                 interpolation=Image.BILINEAR):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size
        self.count = count
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.return_original = return_original

        if isinstance(resize_size, numbers.Number):
            self.resize_size = (int(resize_size), int(resize_size))
        else:
            self.resize_size = resize_size
        self.interp = interpolation        
        self.resizecrop = transforms.Resize(self.resize_size, interpolation=self.interp)

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        img = self._check_size(img)
        if self.return_original:
            originals = []
        results = []
        coords = []
        for i in range(self.count):
            data, coord = self._random_crop(img)
            if self.return_original:
                originals.append(data.copy())
            data = self.resizecrop(data)
            results.append(data)
            coords.append(self._resize_coord(coord))
        if self.return_original:
            return (results, originals, coords)
        else:
            return (results, coords)

    def _check_size(self, x):
        """ Ensures the image is big enough to 
        """
        self.h, self.w = _get_image_size(x)
        # if not using padding boundary for valid crop area, then total size is just crop size
        # if use pad is enforced, there is an extra amount of padding that is not valid, so the resulting image is larger
        total_h = self.crop_size[0]
        total_w = self.crop_size[1]
        if self.h < total_h or self.w < total_w:
            pad_amount = 0
            # calculate image size ratio to preserve preportions after resize
            if self.h < self.w:
                # smaller side will be equal to crop size + pad amount
                ratio_h = 1
                # larger side will be scaled up so that it stays larger
                ratio_w = self.w / self.h
                # unified ratio to increase size by based on smaller side
                ratio_r = total_w / self.h
            else:
                ratio_h = self.h / self.w
                ratio_w = 1
                ratio_r = total_h / self.w
            # do resize based on if either PIL or Tensor
            if _is_pil_image(x):                
                x = x.resize(int(int(self.w * ratio_r) + pad_amount * ratio_w),
                             int(int(self.h * ratio_r) + pad_amount * ratio_h)
                            )
                # get new size
                self.h, self.w = _get_image_size(x)
                return x
            elif isinstance(img, torch.Tensor) and img.dim() > 2:
                x = x.resize(int(int(self.w * ratio_r) + pad_amount * ratio_w),
                             int(int(self.h * ratio_r) + pad_amount * ratio_h)
                            )
                # get new size
                self.h, self.w = _get_image_size(x)
                return x
            else:
                # Numpy? shouldn't happen...
                return x
        else:
            # image is large enough already
            return x

    def _random_crop(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.crop_size[1]:
            img = F.pad(img, (self.crop_size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.crop_size[0]:
            img = F.pad(img, (0, self.crop_size[0] - img.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.crop_size)

        x1 = i
        y1 = j
        x2 = x1 + h
        y2 = y1 + w

        return TF.crop(img, i, j, h, w), (x1, y1, x2, y2, h, w)

    def _resize_coord(self, coord):
        """ Scale the coordinates by the amount the crop was resized
        """
        ratio_x = self.resize_size[0] / self.crop_size[1]
        ratio_y = self.resize_size[0] / self.crop_size[1]
        
        x1 = int(coord[0] * ratio_x)
        y1 = int(coord[1] * ratio_y)
        x2 = int(coord[2] * ratio_x)
        y2 = int(coord[3] * ratio_y)
        h = int(coord[4] * ratio_x)
        w = int(coord[5] * ratio_y)

        return (x1, y1, x2, y2, h, w)


###############################################
## SWAV Style, DirectoryDataset based #######
###########################################

class ContrastiveMultiCropDataset(datasets.ImageFolder):
    def __init__(
        self,
        data_path,
        crop_size,
        resize_size,
        count=5,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ):
        super().__init__(data_path)
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.count = count
        self.crop_pad = crop_pad
        self.use_pad = use_pad

        trans = []
        color_transform = transforms.Compose(
            [get_color_distortion(), RandomGaussianBlur()]
        )

        for i in range(count):
            randomcrop = transforms.RandomCrop(crop_size)
            resizecrop = MultiCropV2(self.resize_size)
            trans.extend(
                [
                    transforms.Compose(
                        [
                            randomcrop,
                            resizecrop,
                            transforms.RandomHorizontalFlip(p=0.5),
                            color_transform,
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean, std=std),
                        ]
                    )
                ]
            )
        self.trans = trans

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops


class RandomGaussianBlur:
    def __call__(self, img):
        do_it = np.random.rand() > 0.5
        if not do_it:
            return img
        sigma = np.random.rand() * 1.9 + 0.1
        return cv2.GaussianBlur(np.asarray(img), (23, 23), sigma)


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort
