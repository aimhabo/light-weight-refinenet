"""RefineNet-LightWeight

RefineNet-LigthWeight PyTorch for non-commercial purposes

Copyright (c) 2018, Vladimir Nekrasov (vladimir.nekrasov@adelaide.edu.au)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import print_function, division

import collections
import glob
import os
import random
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import transforms, utils

class Pad(object):
    """Pad image and mask to the desired size

    Args:
      size (int) : minimum length/width
      img_val (array) : image padding value
      msk_val (int) : mask padding value

    """
    def __init__(self, size, img_val, msk_val, dpt_val=None):
        self.size = size
        self.img_val = img_val
        self.msk_val = msk_val
        if dpt_val is None :
            self.dpt_val = msk_val
        else:
            self.dpt_val = dpt_val

    def __call__(self, sample):
        image, mask, data_type = sample['image'], sample['mask'], sample['data_type']
        h, w = image.shape[:2]
        h_pad = int(np.clip(((self.size - h) + 1)// 2, 0, 1e6))
        w_pad = int(np.clip(((self.size - w) + 1)// 2, 0, 1e6))
        pad = ((h_pad, h_pad), (w_pad, w_pad))
        image = np.stack([np.pad(image[:,:,c], pad,
                         mode='constant',
                         constant_values=self.img_val[c]) for c in range(3)], axis=2)
        mask = np.pad(mask, pad, mode='constant', constant_values=self.msk_val)
        if data_type == 3:
            depth = sample['depth']
            depth = np.pad(depth, pad, mode='constant', constant_values=self.dpt_val)
            return {'image': image, 'mask': mask, 'depth': depth, 'data_type': data_type}
        else:
            return {'image': image, 'mask': mask, 'data_type': data_type}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, crop_size):
        assert isinstance(crop_size, int)
        self.crop_size = crop_size
        if self.crop_size % 2 != 0:
            self.crop_size -= 1

    def __call__(self, sample):
        image, mask, data_type = sample['image'], sample['mask'], sample['data_type']
        h, w = image.shape[:2]
        new_h = min(h, self.crop_size)
        new_w = min(w, self.crop_size)
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        image = image[top: top + new_h,
                        left: left + new_w]
        mask = mask[top: top + new_h,
                    left: left + new_w]
        if data_type == 3:
            depth = sample['depth']
            depth = depth[top: top + new_h,
                    left: left + new_w]
            return {'image': image, 'mask': mask, 'depth': depth, 'data_type': data_type}
        else:
            return {'image': image, 'mask': mask, 'data_type': data_type}

class ResizeShorterScale(object):
    """Resize shorter side to a given value and randomly scale."""

    def __init__(self, shorter_side, low_scale, high_scale):
        assert isinstance(shorter_side, int)
        self.shorter_side = shorter_side
        self.low_scale = low_scale
        self.high_scale = high_scale

    def __call__(self, sample):
        data_type = sample['data_type']
        image, mask = sample['image'], sample['mask']
        min_side = min(image.shape[:2])
        scale = np.random.uniform(self.low_scale, self.high_scale)
        if min_side * scale < self.shorter_side:
            scale = (self.shorter_side * 1. / min_side)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        if data_type == 3:
            depth = sample['depth']
            depth = cv2.resize(depth, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            return {'image': image, 'mask': mask, 'depth': depth, 'data_type': data_type}
        else:
            return {'image': image, 'mask': mask, 'data_type': data_type}

class RandomMirror(object):
    """Randomly flip the image and the mask"""

    def __init__(self):
        pass

    def __call__(self, sample):
        image, mask, data_type = sample['image'], sample['mask'], sample['data_type']
        do_mirror = np.random.randint(2)
        if do_mirror:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if data_type == 3:
            depth = sample['depth']
            if do_mirror:
                depth = cv2.flip(depth, 1)
            return {'image': image, 'mask': mask, 'depth': depth, 'data_type': data_type}
        else:
            return {'image': image, 'mask': mask, 'data_type': data_type}


class Normalise(object):
    """Normalise a tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalise each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, scale, mean, std, classes, depths):
        self.scale = scale
        self.mean = mean
        self.std = std
        self.classes=classes
        self.depths=depths

    def __call__(self, sample):
        image = sample['image']
        image = (self.scale * image - self.mean) / self.std
        mask = sample['mask']
        mmask = np.max(mask)
        mask = mask*self.classes/mmask
        if sample['data_type'] == 3:
            depth = sample['depth']
            mdepth = np.max(depth)
            depth = depth*self.depths/mdepth
            return {'image': image, 'mask' : mask, 'depth': depth, 'data_type': sample['data_type']}
        else:
            return {'image': image, 'mask' : mask, 'data_type': sample['data_type']}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask, data_type = sample['image'], sample['mask'], sample['data_type']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        if data_type == 3:
            depth = sample['depth']
            return {'image': torch.from_numpy(image),
                    'mask': torch.from_numpy(mask),
                    'depth': torch.from_numpy(depth),
                    'data_type': data_type}
        else:
            return {'image': torch.from_numpy(image),
                    'mask': torch.from_numpy(mask),
                    'data_type': data_type}

class NYUDataset(Dataset):
    """NYUv2-40"""

    def __init__(
        self, data_file, transform_trn=None, transform_val=None
        ):
        """
        Args:
            data_file (string): Path to the data file with annotations.
            data_dir (string): Directory with all the images.
            transform_{trn, val} (callable, optional): Optional transform to be applied
                on a sample.
        """

        with open(data_file, 'rb') as f:
            dataline = f.readline()
            self.data_type = dataline.decode('utf-8').strip('\n').count('\t') + 1
        f.close()
        with open(data_file, 'rb') as f:
            datalist = f.readlines()

        if self.data_type == 2:
            self.datalist = [(i, l) for i, l in map(lambda x: x.decode('utf-8').strip('\n').split('\t'), datalist)]
        elif self.data_type == 3:
            self.datalist = [(i,l,d) for i,l,d in map(lambda x: x.decode('utf-8').strip('\n').split('\t'), datalist)]
        else:
            assert ( self.data_type and False)
        self.transform_trn = transform_trn
        self.transform_val = transform_val
        self.stage = 'train'

    def set_stage(self, stage):
        self.stage = stage

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        img_name = self.datalist[idx][0]
        msk_name = self.datalist[idx][1]
        if self.data_type == 3:
            dpt_name = self.datalist[idx][2]
        def read_image(x):
            img_arr = np.array(Image.open(x))
            if len(img_arr.shape) == 2: # grayscale
                img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
            return img_arr
        image = read_image(img_name)
        mask = np.array(Image.open(msk_name))
        if self.data_type == 3:
            depth = np.array(Image.open(dpt_name))

        if self.data_type == 3:
            sample = {'image': image, 'mask': mask, 'depth': depth, 'data_type': self.data_type}
        else:
            sample = {'image': image, 'mask': mask, 'data_type': self.data_type}

        if self.stage == 'train':
            if self.transform_trn:
                sample = self.transform_trn(sample)
        elif self.stage == 'val':
            if self.transform_val:
                sample = self.transform_val(sample)
        return sample
