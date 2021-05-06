import os
import cv2 
import sys
import math
import glob
import time
import random
import collections
import numpy as np
import pandas as pd
import imageio as io
import seaborn as sns
from PIL import Image

from tqdm import tqdm
#from tqdm.notebook import tqdm

from prettytable import PrettyTable

##torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

from torch import Tensor

import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler


##torchvision
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor,Compose
import torchvision.transforms.functional as TF


import torchsummary


# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import clear_output
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')


%load_ext tensorboard


!pip install imgviz
import imgviz


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

def convert_to_color(arr_2d):
    """ Numeric labels to RGB-color encoding """

    palette = {0 : (0, 0, 0), # Impervious surfaces (white)
               1 : (255, 255, 255),     # Buildings (blue)
               0 : (0, 0, 0),   # Low vegetation (cyan)
               0 : (0, 0, 0),     # Trees (green)
               0 : (0, 0, 0),   # Cars (yellow)
               0 : (0, 0, 0),}    # Clutter (red)

    
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)


    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    palette =  {(255, 255, 255): 0, # Impervious surfaces (white)
                (0  , 0  , 255): 1, # Buildings (blue)
                (0  , 255, 255): 0, # Low vegetation (cyan)
                (0  , 255, 0  ): 0, # Trees (green)
                (255, 255, 0  ): 0,   # Cars (yellow)
                (255, 0  , 0  ): 0}    # Clutter (red) # Trees (green)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    plt.axis('off')
    if title is not None:
        plt.title(title)

def poly_lr_scheduler(optimizer,
                      init_lr,
                      iter,
                      lr_decay_iter=1,
                      max_iter=20000,
                      power=0.9):

    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr*(1 - float(iter)/max_iter)**power
    return optimizer

def step_lr_scheduler(optimizer,
                      init_lr,
                      iter,
                      drop=0.1,
                      num_epoch=100,
                      batch_size=40,
                      train_data_num=2000,
                      sample_drop=0.4):
    
    max_iter = (train_data_num//batch_size)* num_epoch
    epochs_drop  = max_iter * sample_drop

    for param_group in optimizer.param_groups:
         param_group['lr'] = init_lr * math.pow(drop, math.floor((1+iter)/epochs_drop))
    return optimizer

def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist

def get_validation_metrics(groundtruth, predicted):
    validation_metrics = {}
    num_classes = 2
    cm_matrix = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predicted, groundtruth):
        cm_matrix += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
        
    overal_accuracy = np.diag(cm_matrix).sum() / cm_matrix.sum()
    
    iu = np.diag(cm_matrix) / (cm_matrix.sum(axis=1) + cm_matrix.sum(axis=0) - np.diag(cm_matrix))
    mean_iu = np.nanmean(iu)
    
    recall = np.diag(cm_matrix) / cm_matrix.sum(axis=1)
    mean_recall = np.nanmean(recall)
    
    precision = np.diag(cm_matrix) / cm_matrix.sum(axis=0)
    mean_precision = np.nanmean(precision)

    F1_Score = (2*precision*recall)/(precision + recall)
    F1_Score_mean = np.mean(F1_Score)

    validation_metrics['Dice_Building'] = F1_Score[1]
    validation_metrics['Dice_mean'] = F1_Score_mean
    validation_metrics['IoU_Building'] = iu[1]
    validation_metrics['MIoU'] = mean_iu
    validation_metrics['Accuracy'] = overal_accuracy
    validation_metrics['Precision'] = mean_precision
    validation_metrics['Recall'] = mean_recall
        
    return validation_metrics

class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist

def plot_total_dsm(dsm_dir, path , name, mode='gray'):

  dsm = io.imread(dsm_dir)

  plt.imshow(dsm, cmap=mode)
  plt.axis('off')

  path_save_result = path + '/' + name + '_' + mode + '.png'

  plt.savefig(path_save_result, dpi=800, bbox_inches='tight', transparent= True, format='png')
  matplotlib.pyplot.close()
  
def plot_orig_image_with_label_method1(image_dir, label_dir, path, name):
  
  im_mask = make_image_with_mask(image_dir, label_dir)
  
  plt.imshow(im_mask)
  plt.axis('off')

  path_save_result = path + '/' + name + '_maskbn' +  '.png'

  plt.savefig(path_save_result, dpi=800, bbox_inches='tight', transparent= True, format='png')
  matplotlib.pyplot.close()

def plot_orig_image_with_label_method2(image_dir, label_dir, path, name):
  
  im_mask = make_rgb_image_with_mask(image_dir, label_dir)
  plt.imshow(im_mask)
  plt.axis('off')

  path_save_result = path + '/' + name + '_maskrgb' + '.png'

  plt.savefig(path_save_result, dpi=800, bbox_inches='tight', transparent= True, format='png')
  matplotlib.pyplot.close()

def plot_total_label(label_dir, path , name):

  mask = cv2.imread(label_dir)
  mask = convert_from_color(mask[..., ::-1])

  plt.imshow(mask,cmap='gray')
  plt.axis('off')

  path_save_result = path + '/' + name + '_label' + '.png'

  plt.savefig(path_save_result, dpi=800, bbox_inches='tight', transparent= True, format='png')
  matplotlib.pyplot.close()

def make_image_with_mask(image_dir,label_dir):

  image = cv2.imread(image_dir)
  
  mask = cv2.imread(label_dir)
  mask = convert_from_color(mask[..., ::-1])

  labelviz_withimg = imgviz.label2rgb(mask, image)

  return labelviz_withimg
  
def make_rgb_image_with_mask(image_dir,label_dir):

  image = cv2.imread(image_dir)
  image = image[..., ::-1]
  image= cv2.add(image,np.array([-25.0]))

  mask = cv2.imread(label_dir)
  mask = convert_from_color(mask[..., ::-1])

  # Create the inverted mask
  mask_inv = cv2.bitwise_not(mask)//255

  # Convert to grayscale image
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  gray= cv2.add(gray,np.array([+25.0]))

  # Extract the dimensions of the original image
  rows, cols, channels = image.shape
  image = image[0:rows, 0:cols]

  # Bitwise-OR mask and original image
  colored_portion = cv2.bitwise_or(image, image, mask = mask)

  # Bitwise-OR inverse mask and grayscale image
  gray_portion = cv2.bitwise_or(gray, gray, mask = mask_inv)
  gray_portion = np.stack((gray_portion,)*3, axis=-1)

  # Combine the two images
  output = colored_portion + gray_portion

  return output

def make_plot_image(label) :
    
    out_label = np.zeros((label.shape[0] , 3 , label[0,:,:].shape[0] ,label[0,:,:].shape[1] ) , dtype=np.uint8)
    
    for i in range(label.shape[0]):
      temp = convert_to_color(label[i,:,:])
      temp = temp.transpose(2,0,1)
      out_label[i,:,:,:] = temp

    return torch.from_numpy(out_label)

def get_confusion_matrix_intersection_mats(groundtruth, predicted):
    """ Returns dict of 4 boolean numpy arrays with True at TP, FP, FN, TN
    """

    confusion_matrix_arrs = {}

    groundtruth_inverse = np.logical_not(groundtruth)
    predicted_inverse = np.logical_not(predicted)

    confusion_matrix_arrs['tp'] = np.logical_and(groundtruth, predicted)
    confusion_matrix_arrs['tn'] = np.logical_and(groundtruth_inverse, predicted_inverse)
    confusion_matrix_arrs['fp'] = np.logical_and(groundtruth_inverse, predicted)
    confusion_matrix_arrs['fn'] = np.logical_and(groundtruth, predicted_inverse)

    return confusion_matrix_arrs   

def image_mask_generator(label,pred):

  W, H = label.shape
  confusion_matrix_colors = {
   'tp': (255, 255, 255),  #cyan
   'fp': (255, 0, 255),  #magenta
   'fn': (0, 255, 255),  #yellow
   'tn': (0, 0, 0)     #black
   }

   
  masks = get_confusion_matrix_intersection_mats(label.ravel(),pred.ravel())
  color_mask = np.zeros((label.ravel().size,3),dtype=np.uint8)

  for label, mask in masks.items():
    color = confusion_matrix_colors[label]
    mask_rgb = np.zeros((W * H ,3),dtype=np.uint8)
    mask_rgb[mask != 0] = color
    color_mask += mask_rgb
  color_mask = color_mask.reshape(W, H, 3)

  return color_mask

def plot_fp_fn(pred , target):

  b, w,h = target.shape

  out_label = np.zeros((b , 3 , w ,h ) , dtype=np.uint8)

  for i in range(b):
    fp_fn = image_mask_generator(target[i],pred[i]).transpose(2,0,1)
    out_label[i,:,:,:] = fp_fn

  return out_label


def ptable_to_csv(table, filename, headers=True):
    #Save PrettyTable results to a CSV file.

    raw = table.get_string()
    data = [tuple(filter(None, map(str.strip, splitline)))
            for line in raw.splitlines()
            for splitline in [line.split('|')] if len(splitline) > 1]
    if table.title is not None:
        data = data[1:]
    if not headers:
        data = data[1:]
    with open(filename, 'w') as f:
        for d in data:
            f.write('{}\n'.format(','.join(d)))

def save_crop_heat(arr, coord, path, name, name_model):

  arr = arr[coord[0]:coord[1],coord[2]:coord[3]] * 255

  path_save_result = path + '/' + name + '_' + name_model + '.png'

  cv2.imwrite(path_save_result, arr)


def save_pred_map(pred_map, path , name, name_model, mode='gray'):
  
  plt.imshow(pred_map, cmap=mode)
  plt.axis('off')

  path_save_result = path + '/' + name + '_' + name_model + '.png'

  plt.savefig(path_save_result, dpi=800, bbox_inches='tight', transparent= True, format='png')
  matplotlib.pyplot.close()

def plot_crop(im, heat, pred, label, coord, path,name):

  fig, axs = plt.subplots(ncols=3)
  fig.tight_layout(pad=-2.5)

  axs[0].imshow(im[coord[0]:coord[1],coord[2]:coord[3],:])
  axs[0].axis('off')

  original_pred = heat[0,0]
  axs[1].imshow(original_pred[coord[0]:coord[1],coord[2]:coord[3]], cmap='jet' )
  axs[1].axis('off')
  
  output_fp_fn = plot_fp_fn(pred.cpu().numpy().astype(np.int) ,label[:,0,...].cpu().numpy())
  axs[2].imshow(output_fp_fn.squeeze(0).transpose(1,2,0)[coord[0]:coord[1],coord[2]:coord[3]] )
  axs[2].axis('off')

  path_save_result = path + '/' + name + '.png'

  plt.savefig(path_save_result, dpi=300, bbox_inches='tight', transparent= True, format='png')
  matplotlib.pyplot.close()

def plot_total(im, heat, pred, label, path, name):

  fig, axs = plt.subplots(ncols=3)
  fig.tight_layout(pad=-2.5)

  axs[0].imshow(im)
  axs[0].axis('off')

  original_pred = heat[0,0]
  axs[1].imshow(original_pred, cmap='jet' )
  axs[1].axis('off')
  
  output_fp_fn = plot_fp_fn(pred.cpu().numpy().astype(np.int) ,label[:,0,...].cpu().numpy())
  axs[2].imshow(output_fp_fn.squeeze(0).transpose(1,2,0))
  axs[2].axis('off')

  path_save_result = path + '/' + name + '.png'

  plt.savefig(path_save_result, dpi=300, bbox_inches='tight', transparent= True, format='png')
  matplotlib.pyplot.close()


def made_image_from_patches(patches_tensor: Tensor, kernel_size: int, stride: int, output_size: tuple, output_original_size: tuple):

    _, C, _, _ = patches_tensor.shape

    kernel_size = kernel_size
    stride = stride
    output_size = output_size

    patches_tensor = patches_tensor.contiguous().unsqueeze(dim=0).permute(0,2,1,3,4).view(1,C,-1,kernel_size*kernel_size)
    patches_tensor = patches_tensor.permute(0, 1, 3, 2) 
    patches_tensor = patches_tensor.contiguous().view(1, C*kernel_size*kernel_size, -1)

    output = F.fold(patches_tensor, output_size=output_size, kernel_size=kernel_size, stride=stride)

    # mask that mimics the original folding:
    recovery_mask = F.fold(torch.ones_like(patches_tensor), output_size=output_size, kernel_size=kernel_size, stride=stride)
    output = output/recovery_mask

    output = output[:,:,0:output_original_size[0],0:output_original_size[1]]

    return output 

def imshow_tensor(data):
  out = data[0].cpu().numpy().transpose(1,2,0)

  return out

def extract_array_patches(array: np.ndarray, kernel_size, stride):

    if len(array.shape) == 3 :
      image = torch.from_numpy(array.transpose(2,0,1))
      image = image.unsqueeze(0)
    
    elif len(array.shape) == 2 :
      image = torch.from_numpy(array)
      image = image.unsqueeze(0).unsqueeze(0)

 
    B, C, H, W = image.shape  # Batch size, here 1, channels (3) if image else 1 if label, height, width

    orig_size = (H // kernel_size) * kernel_size

    patches = image.unfold(3, kernel_size, stride).unfold(2, kernel_size, stride).permute(0,1,2,3,5,4)

    image_patches = patches.contiguous().view(B, C, -1, kernel_size, kernel_size).squeeze(0).permute(1,0,2,3)
    
  
    return image_patches , (orig_size, orig_size)

def make_result(image_dir, label_dir, dsm_dir, kernel_size , stride , model):

  image = Image.open(image_dir)
  label = Image.open(label_dir)
  dsm = Image.open(dsm_dir)

  kernel_size = kernel_size
  stride = stride

  extracted_image_pathces , orig_image_size , output_size = extract_patches(image, kernel_size, stride, True)
  extracted_label_pathces , _               , _           = extract_patches(label, kernel_size, stride, False)  
  extracted_dsm_pathces   , _               , _           = extract_patches(dsm, kernel_size, stride)  

  #print("\n----- input shape extracted_image_pathces & original image size & output_size : ", extracted_image_pathces.shape, orig_image_size,output_size )
  #print("----- input shape extracted_label_pathces & original label size & output_size : ", extracted_label_pathces.shape, orig_image_size,output_size ) 
  #print("----- input shape extracted_dsm_pathces & original dsm size & output_size : ", extracted_dsm_pathces.shape, orig_image_size,output_size ) 

  original_label_from_patches = made_image_from_patches(extracted_label_pathces, kernel_size, stride, output_size, orig_image_size)
  #print("----- original_label_from_patches size : ", original_label_from_patches.shape) 

  B,_,W,H = extracted_label_pathces.shape

  pred_soft_all = torch.zeros_like(torch.rand(B, 2, W, H))

  with torch.no_grad():
    j = 0 
    model.eval()

    for im, lbl, dsm_  in zip(extracted_image_pathces, extracted_label_pathces, extracted_dsm_pathces):
      
      img = im.unsqueeze(dim=0).to(device)
      target  = lbl.unsqueeze(dim=0).to(device)
      dsm = dsm_.unsqueeze(dim=0).to(device)

      pred_soft,_,_ = model(img,dsm)
      #pred_soft = pred_soft1*pred_soft2*pred_soft3

      pred_soft = F.softmax(pred_soft, dim=1)
      
      pred_soft_all[j,:,:,:] = pred_soft
      
      j += 1 

    #Recover_Image
  original_pred_soft = made_image_from_patches(pred_soft_all, kernel_size, stride, output_size, orig_image_size)
  #original_pred_binary_map = (original_pred_soft > 0.5) * 1
  original_pred_binary_map = original_pred_soft.data.max(1)[1]

  return original_label_from_patches, original_pred_binary_map, original_pred_soft  

def extract_patches(array, kernel_size, stride,flag=None):

  if len(np.asarray(array).shape) == 3 :

     if flag == True:
       
       # Transform image to tensor
       array = TF.to_tensor(array)

       # Normalized Data
       normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                      std=[0.5, 0.5, 0.5])
       
       array = normalize(array)
       array = array.unsqueeze(0)  

     else:
      array = torch.from_numpy(convert_from_color(np.array(array))).unsqueeze(0).unsqueeze(0).float()


  elif len(np.asarray(array).shape)  == 2 :
    array = TF.to_tensor(array)
    normalize = transforms.Normalize(mean=[0.5], 
                                      std=[0.5])
       
    array = normalize(array)
    array = array.unsqueeze(0)
  

  kernel_size = kernel_size
  stride = stride  # smaller than kernel will lead to overlap

  array_shape = array.shape
  B, C, H_orig, W_orig = array_shape  # Batch size, channels (3), height, width

  # number of pixels missing in each dimension:
  pad_w = W_orig % stride
  pad_h = H_orig % stride

  #print('pad_w --> ', pad_w)
  #print('pad_h --> ', pad_h)

  array = F.pad(input=array, pad=(0, kernel_size - pad_w, 0, kernel_size - pad_h), mode='constant', value=0)
  array_shape = array.shape

  B, C, H, W = array_shape 
  
  image_patches = array.unfold(3, kernel_size, stride).unfold(2, kernel_size, stride).permute(0,1,2,3,5,4)
  image_patches = image_patches.contiguous().view(B, C, -1, kernel_size, kernel_size).squeeze(0).permute(1,0,2,3)
   
  return image_patches , (H_orig, W_orig) ,(H,W)
  
  
