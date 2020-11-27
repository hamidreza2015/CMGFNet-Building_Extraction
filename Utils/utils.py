
import os
import sys
import math

import collections

import random
import numpy as np
import imageio as io

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torchvision

from prettytable import PrettyTable
from tqdm import tqdm

## Matplotlib
import matplotlib.pyplot as plt


class Logs:
    def __init__(self, path, out=sys.stderr):
        """Create a logs instance on a logs file."""

        self.fp = None
        self.out = out
        if path:
            if not os.path.isdir(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            self.fp = open(path, mode="a")

    def log(self, msg):
        """Log a new message to the opened logs file, and optionnaly on stdout or stderr too."""
        if self.fp:
            self.fp.write(msg + os.linesep)
            self.fp.flush()

        if self.out:
            print(msg, file=self.out)


def sliding_window(image, stride=10, window_size=(20,20)):
    patches = []
    # slide a window across the image
    for x in range(0, image.shape[0], stride):
        for y in range(0, image.shape[1], stride):
            new_patch = image[x:x + window_size[0], y:y + window_size[1]]
            if new_patch.shape[:2] == window_size:
                patches.append(new_patch)
    return patches

def transform(patch, flip=False, mirror=False, rotations=[]):

    transformed_patches = [patch]
    for angle in rotations:
        transformed_patches.append(skimage.img_as_ubyte(skimage.transform.rotate(patch, angle)))
    if flip:
        transformed_patches.append(np.flipud(patch))
    if mirror:
        transformed_patches.append(np.fliplr(patch))
    return transformed_patches

def Image_Extractor(folders ,dataset_dir , dataset_ids , image_size , step_size):

  for suffix, folder, files in tqdm(folders):
    
    os.mkdir(dataset_dir + suffix )
    
    # Generate generators to read the iamges
    test_dataset = (io.imread(folder + files.format(*id_)) for id_ in  dataset_ids)
    
    test_samples = []
    for image in test_dataset:
        # Same as the previous loop, but without data augmentation (test dataset)
        # Sliding window with no overlap
        for patches in sliding_window(image, window_size=image_size, stride=step_size):
            test_samples.extend(transform(patches))

   
    for i, sample in enumerate(test_samples):
        io.imsave('{}/{}.tif'.format(dataset_dir + suffix , i), sample)



def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

     
def get_random_pos(img, window_shape):

  """ Extract of 2D random patch of shape window_shape in the image """
  w, h = window_shape
  W, H = img.shape[-2:]
  x1 = random.randint(0, W - w - 1)
  x2 = x1 + w
  y1 = random.randint(0, H - h - 1)
  y2 = y1 + h

  return x1, x2, y1, y2

class TrainImageLoader(Dataset):
    def __init__(self,
                 ids,
                 image_size,
                 image_files,
                 label_files,
                 train_sample,
                 cache=True,
                 augmentation=True,):
        
        super().__init__()
        
        
        self.augmentation = augmentation
        self.cache = cache
        self.image_size = image_size
        self.train_sample = train_sample


        # List of files
        self.image_files = [image_files.format(*id) for id in ids]
        self.label_files = [label_files.format(*id) for id in ids]

        # Sanity check : raise an error if some files do not exist
        for f in self.image_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))
        
        # Initialize cache dicts
        self.data_cache_ = {}
        self.label_cache_ = {}
           
    
    def __len__(self):
        
        return self.train_sample

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True
        
        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))
            
        return tuple(results)
    
    def __getitem__(self, i):
        
        # Pick a random image
        random_idx = random.randint(0, len(self.image_files) - 1)
        
        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            data = 1/255 * np.asarray(io.imread(self.image_files[random_idx]).transpose((2,0,1)), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data
            
        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else: 
            # Labels are converted from RGB to their numeric values
            label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label
                
   
        # Get a random patch
        x1, x2, y1, y2 = get_random_pos(data, self.image_size)
        


        data_p = data[:, x1:x2,y1:y2]
        label_p = label[x1:x2,y1:y2]
     
        
        # Data augmentation
        data_p, label_p = self.data_augmentation(data_p, label_p)

        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(label_p))

