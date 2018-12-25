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

def combine_classes():
    ''' Lists of combined classes in coco-stuff dataset. '''

    sports= ['34','35','36','37','38','39','40','41','43','42']
    accessory = ['26','27','28','29','30','31','32','33']
    animal = ['16','17','18','19','20','21','22','23','24','25']
    outdoor = ['10','11','12','13','14','15']
    vehicle = ['2','3','4','5','6','7','8','9']
    person = ['1']
    indoor = ['84','85','86','87','88','89','90','91']
    appliance =  ['78','79','80','81','82','83']
    electronic = ['72','73','74','75','76','77']
    furniture = ['62','63','64','65','66','67','68','69','70','71','71','165',
             '156','98','108','66','133','107','130','161','123']
    food = ['52','53','54','55','56','57','58','59','60','61','121','122','170','121']
    kitchen = ['44','45','46','47','48','49','50','51']
    water  = ['148','155','178','179'] #
    ground = ['154','159','111','125','136','140','147','149','126','144','145']#
    solid = ['127','135','162','182','150','160']
    sky = ['106','157']#
    plant = ['163','134','169','94','97','119','129','142']#
    structural = ['146','138','99','113','164']#-
    building = ['128','95','166','151','158','96']#-
    #food = ['121','122','170','121']
    textile = ['152','131','168','137','105','104','109','93','141','92','167']
    #furniture = ['69','71','165','156','98','108','66','133','107','130','161','123']
    window = ['180','181']
    floor = ['114','115','116','117','118']
    ceiling = ['102','103']
    wall = ['171','172','173','174','175','176','177']
    rawmaterial = ['132','143','139','100']
    other = ['120','183']

    others = other+sports+accessory+animal+outdoor+vehicle+person+ \
        indoor+appliance+electronic+furniture+food+kitchen+solid+textile+window+floor+ceiling+wall+rawmaterial
    buildings= structural+building
    return plant, water, ground, sky, buildings, others

def target_transform(crop_size):
    return Compose([CenterCrop(crop_size), ToTensor()])


class Dataset_CocoSegmented(data.Dataset):
    def __init__(self, path, cropsize, annotationFile, no_of_classes,
                 image_extensions=(".png", ".jpg", ".jpeg"), is_train = False, bicubic = False):
        super(Dataset_CocoSegmented, self).__init__()
        self.is_train = is_train
        self.cropsize = cropsize
        self.bicubic = bicubic
        self.root = path
        self.annFile = annotationFile #'/home/wgondal/coco_stuff/stuff_train2017.json'
        self.coco = COCO(self.annFile)
        self.imgIds = self.coco.getImgIds()
        self.num_of_classes = no_of_classes
        self.plant, self.water, self.ground, self.sky, self.buildings, self.others = combine_classes()

    def return_segMask_combinedClasses(self, anns):
        mask = []
        for m in anns:
            if str (m['category_id']) in self.others:
                mask.append(self.coco.annToMask(m)*0)
            elif str(m['category_id']) in self.buildings:
                mask.append(self.coco.annToMask(m)*4)
            elif str(m['category_id']) in self.sky:
                mask.append(self.coco.annToMask(m)*3)
            elif str(m['category_id']) in self.ground:
                mask.append(self.coco.annToMask(m)*2)
            elif str(m['category_id']) in self.water:
                mask.append(self.coco.annToMask(m)*1)
            elif str(m['category_id']) in self.plant:
                mask.append(self.coco.annToMask(m)*5)
        return sum(mask)

    def return_segMask(self, anns):
        '''Return a single mask combining each binary mask'''
        mask = [(self.coco.annToMask(m))*( m['category_id']-1) for m in anns] #coco.annToMask(m)-> mask of 1s and 0s
        return sum(mask)

    def return_labels(self, mask):
        unique_labels = torch.FloatTensor(np.unique(mask))
        labels = torch.zeros(self.num_of_classes)
        labels[unique_labels.long()] = 1
        return labels

    def _scale_and_crop(self,img, seg_mask, cropSize, is_train):
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
            if self.num_of_classes == 183:
                seg = np.full((h,w), 182) #An issue with a few labels, thus returning dummy mask.
            elif self.num_of_classes == 6:
                seg = np.full((h,w), 0)
        h_s, w_s = img.shape[0], img.shape[1]
        x1 = random.randint(0, w_s - cropSize)
        y1 = random.randint(0, h_s - cropSize)

        img_crop = img[y1: y1 + cropSize, x1: x1 + cropSize, :]
        seg_crop = seg[y1: y1 + cropSize, x1: x1 + cropSize]
        return img_crop, seg_crop

    def __getitem__(self, index):
        img_id = self.imgIds[index]
        annIds = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(annIds)
        if self.num_of_classes == 6:
            seg_mask = self.return_segMask_combinedClasses(anns)
        elif self.num_of_classes ==183:
            seg_mask = self.return_segMask(anns)
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = load_img(os.path.join(self.root, path))
        target, scaled_and_cropped_seg_mask = self._scale_and_crop(img, seg_mask, self.cropsize, is_train = True)
        labels = self.return_labels(scaled_and_cropped_seg_mask)
        target = target.astype(np.float32) / 255.
        target = target.transpose((2, 0, 1))
        target = torch.from_numpy(target)
        return target, labels

    def __len__(self):
        return len(self.imgIds)
