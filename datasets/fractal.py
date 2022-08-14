import pathlib
import numbers
import random
from tqdm import tqdm

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .augments import get_blur, get_color_distortion
from .augments import BuildOutput, MultiCropCoord

####################################################################################################################
########### F R A C T A L #############-------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# Custom Fractal Images dataset with high resolution images.  Must apply crop to use.
# ------------------------------------------------------------------------------------------------------------------
class Fractal(Dataset):
    def __init__(self, path="/content/all/", part="train", train_split=0.9, cache=False):
        self.cache = cache
        self.all_data = all_paths = [
            str(p.absolute()) for p in pathlib.Path(path).glob("*")
        ]
        self.total = len(self.all_data)
        self.split = train_split
        if part == "train":
            self.data = self.all_data[: int(self.total * self.split)]

        else:
            self.data = self.all_data[int(self.total * self.split) :]
        if self.cache:
            print(f"Loading {len(self.data)} images into memory...")
            imgs = []
            for p in self.data:
                imgs.append(Image.open(p).convert('RGB'))
            self.data = imgs
            print(f"Cache created with {len(self.data)} elements")
        else:
            print(f"Loading images from disk during training")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.cache:
            return self.transform(self.data[idx])
        else:
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
    transform_list.append(get_color_distortion(s=0.2, use_grayscale=use_grayscale))
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
# Custom Fractal Images dataset with contrastive style batches of the original image and then an augmented version

# ------------------------------------------------------------------------------------------------------------------
# Custom Fractal Images dataset with high resolution images.  Must apply crop and also return the coordinates of
# of the crop in the form of the upper left and lower right points (bounding box - [x1,y1,x2,y2]). Supports the concept
# of multiple crops from each image so that Contrastive Learning can be used with each crop from the same image has a label
# applied in this class based on the index
# ------------------------------------------------------------------------------------------------------------------
class FractalLabel(Dataset):
    def __init__(self, path="/content/all/", part="train", train_split=0.9, cache=False):
        self.cache = cache
        self.all_data = all_paths = [
            str(p.absolute()) for p in pathlib.Path(path).glob("*")
        ]
        self.total = len(self.all_data)
        self.split = train_split
        if part == "train":
            self.data = self.all_data[: int(self.total * self.split)]

        else:
            self.data = self.all_data[int(self.total * self.split) :]
        if self.cache:
            print(f"Loading {len(self.data)} images into memory...")
            imgs = []
            for p in self.data:
                imgs.append(Image.open(p).convert('RGB'))
            self.data = imgs
            print(f"Cache created with {len(self.data)} elements")
        else:
            print(f"Loading images from disk during training")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.cache:
            result, coords = self.transform(self.data[idx])
        else:
            result, coords = self.transform(Image.open(self.data[idx]).convert("RGB"))
        label = torch.full((result.shape[0],), fill_value=idx, dtype=torch.int)
        return (result, label, coords)


# ------------------------------------------------------------------------------------------------------------------
# Prepares a set of transformations that makes many crops of a certain scale square area randomly from each image
# in a batch, effectively making a much larger dataset than individual image count suggests. Also returns the coordinates
# of each crop.----- Results in a 4-d tensor [N, C, H, W] with N being number of crops
# -------------------------------------------------------------------------------------------------------------
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
    transform_list.append(get_color_distortion(s=0.2, use_grayscale=False))
    transform_list.append(MultiCropCoord(crop_size, image_size, count=crop_mode))
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
    transform_list.append(transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.3, saturation=0.3, hue=0.2)], 0.7))
    transform_list.append(MultiCropCoord(crop_size, image_size, count=crop_mode, return_original=True))
    transform_list.append(BuildOutput(mean, std, super_res=True))

    dataset.transform = transforms.Compose(transform_list)
    return DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
    )


###############################################
## SWAV Style, DirectoryDataset based #######
###########################################

class ContrastiveMultiCropDataset(Dataset):
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
        color_transform = transforms.Compose([transforms.ColorJitter(0.5,0.5,0.2), get_blur()])

        for i in range(count):
            randomcrop = transforms.RandomCrop(crop_size)
            resizecrop = MultiCropCoord(self.resize_size)
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
###############################################
## TUNIT Style, Basic Contrastive output but includes label and coordinates #######
###########################################

class FractalTUNITContrastive(Dataset):
    def __init__(self, path="/content/all/", part="train"):
        self.all_data = all_paths = [
            str(p.absolute()) for p in pathlib.Path(path).glob("*")
        ]
        self.total = len(self.all_data)
        if part == "train":
            self.data = self.all_data[: int(self.total * 0.9)]
        elif part == "all":
            self.data = self.all_data
        else:
            self.data = self.all_data[int(self.total * 0.9) :]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_img = Image.open(self.data[idx]).convert("RGB")
        # Get Augmented version of image
        result, original, coords = self.transform(raw_img)
        label = torch.full((result.shape[0],), fill_value=idx, dtype=torch.int)
        transform_output = (result, original, coords, label)
        # Get version of image that is only cropped, no other augs
        result, original, coords = self.crop(raw_img)
        label = torch.full((result.shape[0],), fill_value=idx, dtype=torch.int)
        crop_output = (result, original, coords, label)

        return transform_output, crop_output

def make_fractal_TUNIT_dataloader(
    dataset,
    batch_size,
    image_size=4,
    crop_size=512,
    num_workers=3,
    aug_strength=0.8,
    use_grayscale=False,
    crop_mode="random",
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
):
    s = aug_strength
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
    transform_list.append(get_blur(p=0.6, s=s, kernel_size=17))
    transform_list.append(get_color_distortion(s=0.6, use_grayscale=use_grayscale))
    if isinstance(crop_mode, int):
        transform_list.append(MultiCropCoord(crop_size, image_size, count=crop_mode, return_original=True))
        transform_list.append(BuildOutput(mean, std, super_res=True))
    else:
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean, std, inplace=True))

    dataset.transform = transforms.Compose(transform_list)

    crop_list = []
    if isinstance(crop_mode, str):
        if crop_mode == "random":
            crop_list.append(
                transforms.RandomCrop(
                    crop_size, pad_if_needed=True, padding_mode="symmetric"
                )
            )
        elif crop_mode == "center":
            crop_list.append(transforms.CenterCrop(crop_size))
        crop_list.append(transforms.Resize((image_size, image_size)))
    crop_list.append(get_blur(p=0.3, s=s, kernel_size=24))
    if isinstance(crop_mode, int):
        crop_list.append(MultiCropCoord(crop_size, image_size, count=crop_mode, return_original=True))
        crop_list.append(BuildOutput(mean, std, super_res=True))
    else:
        crop_list.append(transforms.ToTensor())
        crop_list.append(transforms.Normalize(mean, std, inplace=True))

    dataset.crop = transforms.Compose(crop_list)

    return DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
