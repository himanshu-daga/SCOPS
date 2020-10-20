# SOFA 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import random
import matplotlib.pyplot as plt
import collections
import torchvision
from PIL import Image

import os.path as osp
import numpy as np

import scipy.io as sio
from absl import flags, app

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

# from . import base as base_data
# from utils import transformations

# -------------- flags ------------- #
# kData = '/home/ubuntu/work/scops/SCOPS/data/CUB_200_2011'
# cub_cache_dir = '/home/ubuntu/work/scops/SCOPS/data/nmr/cachedir/cub'
sofa_dir = '/home/ubuntu/work/scops/SCOPS/data/CUB_sofa'
IMG_MEAN = np.array((122.67891434, 116.66876762,
                     104.00698793), dtype=np.float32)
# -------------- Dataset ------------- #
class SofaDataset(Dataset):
    '''
    Sofa Data loader
    '''
    def __init__(self, opts, filter_key=None):
        # super(SofaDataset, self).__init__(opts, filter_key=filter_key)
        self.data_dir = osp.join(sofa_dir, 'images')
        self.img_list_path = osp.join(sofa_dir, 'images.txt')
        self.crop_h, self.crop_w = (256,256)

        # Load the list of images
        print('loading %s' % self.img_list_path)
        with open(self.img_list_path) as f:
            img_list = f.read().splitlines()
        self.files = []
        for name in img_list:
            self.files.append({
                'img': osp.join(self.data_dir, name),
                'name':name
            })
        print('%d images' % len(self.files))
    
    def __len__(self):
        return len(self.files)

    def generate_scale_imgs(self, img, interp_mode):
        f_scale_y = self.crop_h/img[0].shape[0]
        f_scale_x = self.crop_w/img[0].shape[1]
        self.scale_y, self.scale_x = f_scale_y, f_scale_x
        if img is not None:
            # img = cv2.resize(img, None, fx=f_scale_x, fy=f_scale_y)
            img = cv2.resize(img, (256,256))
        return img

    def __getitem__(self, index):
        datafile = self.files[index]
        # print('HERE: ',datafile)
        image = cv2.imread(datafile["img"], cv2.IMREAD_COLOR)
        name = datafile["name"]
        # label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        label = None
        if label is not None:
            label = label.astype(np.float32)
            label /= 255.0
            label = cv2.resize(label, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_LINEAR)

        # always scale to fix size
        # print('Size of',name,' b4 scale ',image.shape)
        image = self.generate_scale_imgs(image, cv2.INTER_LINEAR)
        # print('Size of',name,' after scale ',image.shape)
        image = np.asarray(image, np.float32)
        image -= IMG_MEAN
        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))

        data_dict = {'img'     : image.copy(),
                     'name'    : name}
        return data_dict

#----------- Data Loader ----------#
def data_loader(opts, shuffle=True):
    dset = SofaDataset(opts)
    return DataLoader(dset,
            batch_size=opts.batch_size,
            shuffle=shuffle,
            num_workers=4,
            drop_last=True)
