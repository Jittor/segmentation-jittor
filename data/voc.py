# ***************************************************************
# Copyright(c) 2019
#     Meng-Hao Guo <guomenghao1997@gmail.com>
#     Dun Liang <randonlang@gmail.com>.
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from jittor.dataset.dataset import Dataset, dataset_root
import jittor as jt
import os
import os.path as osp
from PIL import Image, ImageOps, ImageFilter

import numpy as np
import scipy.io as sio

import random
import settings

def fetch(image_path, label_path=None):
    with open(image_path, 'rb') as fp:
        image = Image.open(fp).convert('RGB')

    with open(label_path, 'rb') as fp:
        label = Image.open(fp).convert('P')

    return image, label


def scale(image, label=None):
    SCALES = settings.SCALES
    ratio = np.random.choice(SCALES)
    w,h = image.size
    nw = (int)(w*ratio)
    nh = (int)(h*ratio)

    image = image.resize((nw, nh), Image.BILINEAR)
    label = label.resize((nw, nh), Image.NEAREST)

    return image, label


def pad(image, label=None):
    w,h = image.size

    crop_size = settings.CROP_SIZE
    pad_h = max(crop_size - h, 0)
    pad_w = max(crop_size - w, 0)
    if pad_h > 0 or pad_w > 0:
        image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
        label = ImageOps.expand(label, border=(0, 0, pad_w, pad_h), fill=settings.IGNORE_INDEX)

    return image, label

def pad_inf(image, label):
    w, h = image.size 
    stride = settings.STRIDE
    pad_h = (stride + 1 - h % stride) % stride
    pad_w = (stride + 1 - w % stride) % stride
    if pad_h > 0 or pad_w > 0:
        image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
        label = ImageOps.expand(label, border=(0, 0, pad_w, pad_h), fill=settings.IGNORE_INDEX)
    
    return image, label

def crop(image, label=None):
    w, h = image.size
    crop_size = settings.CROP_SIZE
    x1 = random.randint(0, w - crop_size)
    y1 = random.randint(0, h - crop_size)
    image = image.crop((x1, y1, x1 + crop_size, y1 + crop_size))
    label = label.crop((x1, y1, x1 + crop_size, y1 + crop_size))


    return image, label


def normalize(image, label):
    mean = (0.485, 0.456, 0.40)
    std = (0.229, 0.224, 0.225)
    image = np.array(image).astype(np.float32)
    label = np.array(label).astype(np.float32)

    image /= 255.0
    image -= mean
    image /= std
    return image, label


def flip(image, label=None):
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return image, label


class BaseDataset(Dataset):
    def __init__(self,  data_root=dataset_root+'/voc/', split='train', batch_size=1, shuffle=False):
        super().__init__()
        ''' total_len , batch_size, shuffle must be set '''
        self.data_root = data_root
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.image_root = os.path.join(data_root, 'images')
        self.label_root = os.path.join(data_root, 'annotations')

        self.data_list_path = os.path.join(self.data_root, self.split + '.txt')
        self.image_path = []
        self.label_path = []

        with open(self.data_list_path, "r") as f:
            lines = f.read().splitlines()

        for idx, line in enumerate(lines):
            _img_path = os.path.join(self.image_root, line + '.jpg')
            _label_path = os.path.join(self.label_root, line + '.png')

            assert os.path.isfile(_img_path)
            assert os.path.isfile(_label_path)
            self.image_path.append(_img_path)
            self.label_path.append(_label_path)
        self.total_len = len(self.image_path)

        # set_attr must be called to set batch size total len and shuffle like __len__ function in pytorch
        self.set_attrs(batch_size = self.batch_size, total_len = self.total_len, shuffle = self.shuffle) # bs , total_len, shuffle


    def __getitem__(self, image_idimage_id):
        return NotImplementedError


class TrainDataset(BaseDataset):
    def __init__(self,  data_root=dataset_root+'/voc/', split='train', batch_size=1, shuffle=False):
        super(TrainDataset, self).__init__(data_root, split, batch_size, shuffle)

    def __getitem__(self, image_id):
        image_path = self.image_path[image_id]
        label_path = self.label_path[image_id]
        image, label = fetch(image_path, label_path)
        image, label = scale(image, label)
        image, label = pad(image, label)
        image, label = crop(image, label)
        image, label = flip(image, label)
        image, label = normalize(image, label)

        image = np.array(image).astype(np.float).transpose(2, 0, 1)
        image = jt.array(image)
        label = jt.array(np.array(label).astype(np.int))

        return image, label


class ValDataset(BaseDataset):
    def __init__(self,  data_root=dataset_root+'/voc/', split='train', batch_size=1, shuffle=False):
        super(ValDataset, self).__init__(data_root, split, batch_size, shuffle)

    def __getitem__(self, image_id):
        image_path = self.image_path[image_id]
        label_path = self.label_path[image_id]

        image, label = fetch(image_path, label_path)
        image, label = pad_inf(image, label)

        image, label = normalize(image, label)

        image = np.array(image).astype(np.float).transpose(2, 0, 1)
        image = jt.array(image)
        label = jt.array(np.array(label).astype(np.int))

        return image, label




def test_dt():


    train_loader = TrainDataset(data_root='/vocdata/', split='train', batch_size=4, shuffle=True)
    val_loader = ValDataset(data_root='/vocdata/', split='val', batch_size=1, shuffle=False)
    print('train', len(train_loader))
    for idx, (image, target) in enumerate(train_loader):
        if idx > 5 :
            break
        print (image.shape, target.shape)

    for idx, (image, target) in enumerate(val_loader):
        if idx > 5 :
            break
        print (image.shape, target.shape)



if __name__ == '__main__':
    test_dt()
