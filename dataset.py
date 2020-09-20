import os
import logging
import pathlib
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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
        return self.transform(Image.open(self.data[idx]).convert('RGB'))
# ------------------------------------------------------------------------------------------------------------------
# Prepares a set of transformations that crops a certain scale square area randomly from each images
# in a batch, effectively making a much larger dataset than individual image count suggests.
# ------------------------------------------------------------------------------------------------------------------
def make_fractal_alae_dataloader(dataset, batch_size, image_size=4, crop_size=512, 
num_workers=3, crop_mode='random', mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    transform_list = []
    if isinstance(crop_mode, str):
        if crop_mode == 'random':
            transform_list.append(transforms.RandomCrop(crop_size, pad_if_needed=True, padding_mode='symmetric'))        
        elif crop_mode == 'center':
            transform_list.append(transforms.CenterCrop(crop_size))                
    transform_list.append(transforms.Resize((image_size, image_size)))            
    transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    transform_list.append(transforms.RandomVerticalFlip(p=0.5))
    transform_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.3, saturation=0.3, hue=0.2))   
    transform_list.append(transforms.RandomGrayscale(p=0.1)) 
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
    transform_list.append(transforms.RandomGrayscale(p=0.1))     
    transform_list.append(MultiCrop(crop_size, image_size, count=crop_mode))
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

class MultiCrop:
    def __init__(self, crop_size, resize_size, count=5, crop_pad=0.1):
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.count = count
        self.crop_pad = crop_pad

    def __call__(self, x):
        x = self._check_size(x)
        results = []
        coords = []
        for i in range(self.count):
            data, coord = self._random_crop(x, i)
            results.append(data)
            coords.append(coord)

        return (self._resize(results), coords)

    def _check_size(self, x):
        self.h, self.w = _get_image_size(x)
        if self.h < (self.crop_size + (self.h * self.crop_pad)) or self.w < (self.crop_size + (self.w * self.crop_pad)):
            if _is_pil_image(x):
                return x.resize((int(self.w * 2), int(self.h * 2)))
            elif isinstance(img, torch.Tensor) and img.dim() > 2:
                return x.resize((int(self.w * 2), int(self.h * 2)))
            else:
                return x
        else:
            return x

    def _random_crop(self, x, i):
        # get total height and width of crop
        if isinstance(self.crop_size, int):
            th, tw = self.crop_size, self.crop_size
        elif isinstance(self.crop_size, float):
            th, tw = int(self.crop_size), int(self.crop_size)
        else:
            th, tw = int(self.crop_size[0]), int(self.crop_size[1])
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
        # calculate available space left over after crop and padding (max x/y)
        available_h = self.h - th - ph
        available_w = self.w - tw - pw  
        padding_h = padding_w = 0         
        if available_h < th:
            # this much extra room needed in height
            padding_h = th - available_h            
        if available_w < tw:
            # this many extra pixels needed in width
            padding_w = tw - available_w
        available_h += padding_h
        available_w += padding_w

        if available_h > 0 and available_h > pw:
            mod_h = random.randint(pw, available_h)
        if available_w > 0 and available_w > ph:
            mod_w = random.randint(ph, available_w)

        x1, y1, x2, y2 = mod_h, mod_w, mod_h + th - padding_h, mod_w + tw - padding_w
        return TF.crop(x, x1, y1, x2, y2), (x1, y1, x2, y2)

    def _resize(self, results):
        resized = []
        for result in results:
            resized.append(result.resize((self.resize_size, self.resize_size)))
        return resized

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
