#!/usr/bin/env python3
"""Loads image datasets from folders of images."""

from PIL import Image
from os import listdir
from os.path import join
import torch.utils.data as data
import numpy as np
import random
from scipy.misc import imread, imresize
import torch
from pycocotools.coco import COCO
import os
from torchvision.transforms import Compose, CenterCrop, ToTensor

def is_image_file(filename, extensions):
    return any(filename.lower().endswith(ext) for ext in extensions)

def load_img(filepath):
    return Image.open(filepath).convert('RGB')

def _scale_and_crop(img, seg_mask, cropSize, is_train):
    h, w = np.size(img,0), np.size(img,1)
    if is_train:
        # random scale
        scale = random.random() + 0.2     # 0.5-1.5
        scale = max(scale, 1. * cropSize / (min(h, w) - 1))
    else:
        # scale to crop size
        scale = 1. * cropSize / (min(h, w) - 1)

    img = imresize(img, scale, interp='bicubic')
    try:
        seg = imresize(seg_mask, scale, interp='nearest')
    except:
        seg = np.full((h,w), 182) #An issue with one image, thus returning dummy mask.
    h_s, w_s = img.shape[0], img.shape[1]
    x1 = random.randint(0, w_s - cropSize)
    y1 = random.randint(0, h_s - cropSize)

    img_crop = img[y1: y1 + cropSize, x1: x1 + cropSize, :]
    seg_crop = seg[y1: y1 + cropSize, x1: x1 + cropSize]
    return img_crop, seg_crop


def return_segMask(anns, coco):
    '''Return 6 segmentaion masks sorted on the basis of their area. The 7th mask belongs to others class'''
    mask = [(coco.annToMask(m))*( m['category_id']-1) for m in anns] #coco.annToMask(m)-> mask of 1s and 0s
    return sum(mask)

def target_transform(crop_size):
    return Compose([CenterCrop(crop_size), ToTensor()])

def return_labels(mask, num_of_classes):
    unique_labels = torch.FloatTensor(np.unique(mask))
    labels = torch.zeros(num_of_classes)
    labels[unique_labels.long()] = 1
    return labels


class Dataset_CocoSegmented(data.Dataset):
    def __init__(self, path, cropsize, annotationFile,
                 image_extensions=(".png", ".jpg", ".jpeg"), is_train = False, bicubic = False):
        super(Dataset_CocoSegmented, self).__init__()
        self.is_train = is_train
        self.cropsize = cropsize
        self.bicubic = bicubic
        self.root = path
        self.annFile = annotationFile #'/home/wgondal/coco_stuff/stuff_train2017.json'
        self.coco = COCO(self.annFile)
        self.imgIds = self.coco.getImgIds()

    def __getitem__(self, index):
        img_id = self.imgIds[index]
        annIds = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(annIds)
        seg_mask = return_segMask(anns, self.coco)
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = load_img(os.path.join(self.root, path))
        target, scaled_and_cropped_seg_mask = _scale_and_crop(img, seg_mask, self.cropsize, is_train = True)

        labels = return_labels(scaled_and_cropped_seg_mask, 183)
        target = target.astype(np.float32) / 255.
        target = target.transpose((2, 0, 1))
        target = torch.from_numpy(target)
        return target, labels

    def __len__(self):
        return len(self.imgIds)
