import os
import logging
import pathlib
import random
import numbers
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None

from os.path import splitext
from os import listdir
from glob import glob

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def _is_numpy(img):
    return isinstance(img, np.ndarray)

def _is_numpy_image(img):
    return img.ndim in {2, 3}


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.imgs_path = pathlib.Path(imgs_dir)
        self.masks_dir = masks_dir
        self.masks_path = pathlib.Path(masks_path)
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [p.stem for p in self.imgs_path.iterdir()
                    if not p.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}

####################################################################################################################
########### C E L E B A #############-------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# CelebA face image dataset, only returns images and not metadata
# ------------------------------------------------------------------------------------------------------------------
class CelebA(Dataset):
    def __init__(self, path='/root/data/CelebA/img_align_celeba/', part='train'):
        if part=='train':
            self.data = [os.path.join(path, file) for file in os.listdir(path)][:182637]
        else:
            self.data = [os.path.join(path, file) for file in os.listdir(path)][182637:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.data[idx]))

def make_celeba_dataloader(dataset, batch_size, image_size=4):
    dataset.transform = transforms.Compose([
                                            transforms.Resize((image_size, image_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])
    return DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=True)

####################################################################################################################
########### F R A C T A L #############-------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# Custom Fractal Images dataset with high resolution images.  Must apply crop to use.
# ------------------------------------------------------------------------------------------------------------------
class Fractal(Dataset):
    def __init__(self, path='/content/all/', part='all', cache='memory'):
        self.all_data = [str(p.absolute()) for p in pathlib.Path(path).glob("*")]
        self.total = len(self.all_data)
        if part == 'all':
            self.data = self.all_data
        elif part=='train':
            self.data = self.all_data[:int(self.total*0.9)]
        else:
            self.data = self.all_data[int(self.total*0.9):]

        self.cache = cache
        if self.cache == 'memory':
            logging.info(f"Using in memory cache for {self.total} images")
            cache_temp = []
            for p in tqdm(self.data):
                try:
                    cache_temp.append(Image.open(p).convert('RGB'))
                except Exception as e:
                    logging.error(f"Failed loading image in dataset:\n{e}")
            self.data = cache_temp
            del cache_temp


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.data[idx]).convert('RGB'))
# ------------------------------------------------------------------------------------------------------------------
# Prepares a set of transformations that crops a certain scale square area randomly from each images
# in a batch, effectively making a much larger dataset than individual image count suggests.
# ------------------------------------------------------------------------------------------------------------------
def make_fractal_alae_dataloader(dataset, batch_size, 
                                 image_size=4,
                                 crop_size=512,
                                 num_workers=3,
                                 crop_mode='random',
                                 mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5),
                                 jitter_settings={"brightness": 0.1, 
                                                  "contrast": 0.3, 
                                                  "saturation": 0.3, 
                                                  "hue": 0.3}):
    transform_list = []
    if isinstance(crop_mode, str):
        if crop_mode == 'random':
            transform_list.append(transforms.RandomCrop(crop_size,
                                                        pad_if_needed=True,
                                                        padding_mode='symmetric'))
        elif crop_mode == 'center':
            transform_list.append(transforms.CenterCrop(crop_size))
    transform_list.append(transforms.Resize((image_size, image_size)))
    transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    transform_list.append(transforms.RandomVerticalFlip(p=0.5))
    transform_list.append(transforms.ColorJitter(**jitter_settings))
    #transform_list.append(transforms.RandomGrayscale(p=0.1))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean, std, inplace=True))

    dataset.transform = transforms.Compose(transform_list)
    return DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, drop_last=True)

# ------------------------------------------------------------------------------------------------------------------
# Custom Fractal Images dataset with high resolution images.  Must apply crop and also return the coordinates of
# of the crop in the form of the upper left and lower right points (bounding box - [x1,y1,x2,y2]). Supports the concept
# of multiple crops from each image so that Contrastive Learning can be used with each crop from the same image has a label
# applied in this class based on the index
# ------------------------------------------------------------------------------------------------------------------
class FractalLabel(Dataset):
    def __init__(self, path='/content/all/', part='train'):
        self.all_data = all_paths = [str(p.absolute()) for p in pathlib.Path(path).glob("*")]
        self.total = len(self.all_data)
        if part=='train':
            self.data = self.all_data[:int(self.total*0.9)]
        else:
            self.data = self.all_data[int(self.total*0.9):]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        result, coords = self.transform(Image.open(self.data[idx]).convert('RGB'))
        label = torch.full((result.shape[0],), fill_value=idx, dtype=torch.int)
        return (result, label, coords)

# ------------------------------------------------------------------------------------------------------------------
# Prepares a set of transformations that makes many crops of a certain scale square area randomly from each image
# in a batch, effectively making a much larger dataset than individual image count suggests. Also returns the coordinates
# of each crop. Results in a 4-d tensor [N, C, H, W] with N being number of crops
# ------------------------------------------------------------------------------------------------------------------
def make_fractal_clr_dataloader(dataset, batch_size, image_size=4, crop_size=512, num_workers=3, crop_mode=5, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    transform_list = []
    transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    transform_list.append(transforms.RandomVerticalFlip(p=0.5))
    transform_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.3, saturation=0.3, hue=0.2))
    #transform_list.append(transforms.RandomGrayscale(p=0.1))
    transform_list.append(MultiCropCoordV2(crop_size, image_size, count=crop_mode))
    transform_list.append(BuildOutput(mean, std))

    dataset.transform = transforms.Compose(transform_list)
    return DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, drop_last=True)



def _get_image_size(img):
    if _is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))

class MultiCropCoord:
    def __init__(self, crop_size, resize_size,
                count=5,
                crop_pad=0.,
                use_pad=False,
                seed=42):
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.count = count
        self.crop_pad = crop_pad
        self.use_pad = use_pad
        self.seed = seed

    def __call__(self, x):
        x = self._check_size(x)
        results = []
        coords = []
        for i in range(self.count):
            data, coord = self._random_crop(x)
            results.append(data)
            coords.append(coord)

        return (self._resize_img(results), self._resize_coords(coords))

    def _check_size(self, x):
        """ Ensures the image is big enough to
        """
        self.h, self.w = _get_image_size(x)
        # if not using padding boundary for valid crop area, then total size is just crop size
        # if use pad is enforced, there is an extra amount of padding that is not valid, so the resulting image is larger
        total_h = self.crop_size + (self.h * self.crop_pad) if self.use_pad else self.crop_size
        total_w = self.crop_size + (self.w * self.crop_pad) if self.use_pad else self.crop_size
        if self.h < total_h or self.w < total_w:
            pad_amount = int(self.crop_size * self.crop_pad)
            # calculate image size ratio to preserve preportions after resize
            if self.h < self.w:
                # smaller side will be equal to crop size + pad amount
                ratio_h = 1
                # larger side will be scaled up so that it stays larger
                ratio_w = self.w / self.h
                # unified ratio to increase size by based on smaller side
                ratio_r = self.crop_size / self.h
            else:
                ratio_h = self.h / self.w
                ratio_w = 1
                ratio_r = self.crop_size / self.w
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

    def _random_crop(self, x):
        # get total height and width of crop
        if isinstance(self.crop_size, int):
            th, tw = self.crop_size, self.crop_size
        elif isinstance(self.crop_size, float):
            th, tw = int(self.crop_size), int(self.crop_size)
        else:
            th, tw = int(self.crop_size[0]), int(self.crop_size[1])
        if self.use_pad:
            # calculate ratio to modify padding by to make it balanced on rectangles
            if self.h < self.w:
                ratio_h = self.h / self.w
                ratio_w = 1.
            else:
                ratio_w = self.w / self.h
                ratio_h = 1.
            # calculate padding to ensure no overlap with corners
            ph = int(self.h * self.crop_pad * ratio_h)
            pw = int(self.w * self.crop_pad * ratio_w)
        else:
            ph = pw = 0
        # calculate available space left over after crop and padding (max x/y)
        available_h = self.h - th - ph
        available_w = self.w - tw - pw
        padding_h = padding_w = 0
        if available_h < 0:
            # this much extra room needed in height
            padding_h = abs(available_h)
        if available_w < 0:
            # this many extra pixels needed in width
            padding_w = abs(available_w)
        available_h += padding_h
        available_w += padding_w

        if available_h > 0 and available_h > pw:
            mod_h = random.randint(pw, available_h)
        else:
            diff = pw - available_h
            mod_h = random.randint(available_h-diff, available_h)
        if available_w > 0 and available_w > ph:
            mod_w = random.randint(ph, available_w)
        else:
            diff = ph - available_w
            mod_w = random.randint(available_w-diff, available_w)

        x1, y1, x2, y2 = mod_h, mod_w, mod_h + th - padding_h, mod_w + tw - padding_w
        # torchvision.transforms.functional.crop(img, top, left, height, width)
        #return TF.crop(x, y1, x1, abs(y2-y1), abs(x2-x1)), (x1, y1, x2, y2, self.h, self.w)
        return transforms.RandomResizedCrop(
                self.crop_size,
                scale=(self.resize_size, self.resize_size),
            )(x)

    def _resize_img(self, results):
        resized = []
        for result in results:
            resized.append(result.resize((self.resize_size, self.resize_size)))
        return resized

    def _resize_coords(self, coords):
        """ Scale the coordinates by the amount the crop was resized
        """
        resized = []
        for coord in coords:
            ratio = self.resize_size / self.crop_size

            x1 = int(coord[0] * ratio)
            y1 = int(coord[1] * ratio)
            x2 = int(coord[2] * ratio)
            y2 = int(coord[3] * ratio)
            h = int(coord[4] * ratio)
            w = int(coord[5] * ratio)
            resized.append((x1, y1, x2, y2, h, w))
        return resized

class MultiCropCoordV2(object):
    def __init__(self, crop_size, resize_size,
                 count=5,
                 padding=None,
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

class BuildOutput:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, x):
        y = x[1]
        x = x[0]
        data = torch.stack([transforms.Normalize(self.mean, self.std, inplace=True)(
            torch.from_numpy(np.array(crop, np.float32, copy=False).transpose((2, 0, 1))).contiguous()) for crop in x])
        label = torch.Tensor(y)

        return data, label

# ------------------------------------------------------------------------------------------------------------------
# MultiCropDataset from SWAV that makes multiple crops of various sizes - close, but we want all the same size
# ------------------------------------------------------------------------------------------------------------------

class MultiCropDataset(datasets.ImageFolder):
    def __init__(
        self,
        data_path,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        return_index=False,
        mean=[0.485, 0.456, 0.406],
        std=[0.228, 0.224, 0.225]
    ):
        super().__init__(data_path)
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        trans = []
        color_transform = transforms.Compose([get_color_distortion(), RandomGaussianBlur()])

        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                color_transform,
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.trans = trans

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops


class RandomGaussianBlur(object):
    def __call__(self, img):
        do_it = np.random.rand() > 0.5
        if not do_it:
            return img
        sigma = np.random.rand() * 1.9 + 0.1
        return cv2.GaussianBlur(np.asarray(img), (23, 23), sigma)


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort
