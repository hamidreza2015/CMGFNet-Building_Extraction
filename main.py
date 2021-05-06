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

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = xm.xla_device()
  

dataset_name = "Vaihingen" 
max_epoch = 30 #@param {type: "number"}
batch_size =   16#@param {type: "number"}
learning_rate =  0.001  #@param {type: "number"}
optimization = "Adamax" #@param ["SGD", "Adam","Adadelta", "Adagrad" ,"Adamax", "AdamW", "ASGD","RMSProp","RProp" ] 
lr_power = 0.3 #@param {type: "number"}
image_size =  256 #@param {type: "number"}
model_name = 'CMGFNet_withGate_loss_08'  #@param {type: "string"}
loss_function_criteria = 'BCE_loss'  #@param {type: "string"}
colab_account_position = 'phdthesis2021@gmail.com'

description = """fusion_model based on cross_gated propossed method """


config = {}
config['dataset_name'] = dataset_name
config['max_epoch'] = max_epoch
config['start_epoch'] = 1            
config['batch_size'] = batch_size
config['learning_rate'] =learning_rate
config['image_size'] = (image_size,image_size)
config['model_name'] = model_name
config['RandomSeed'] = 1234
config['optimization'] = optimization
config['lr_power'] = lr_power


log_path = '/content/drive/My Drive/runs/' + config['model_name']

#log = Logs(os.path.join(log_path, 'log_' + config['model_name']))



path_orig = '/content/sample2/'

if not os.path.isdir(path_orig):
    os.mkdir(path_orig)



path_img =  path_orig + "images"
if not os.path.isdir(path_img):

  os.mkdir(path_img)

path_lbl = path_orig + "labels"
if not os.path.isdir(path_lbl):

  os.mkdir(path_lbl)

path_dsm = path_orig + "dsms"
if not os.path.isdir(path_dsm):

  os.mkdir(path_dsm)

 crop_size = [600, 640, 680, 720, 840] #640+
#crop_size = [256] #640+

counter = 1
for i in range(len(label_dir)):


  image = cv2.imread(image_dir[i],cv2.COLOR_BGR2RGB)
  label = cv2.imread(label_dir[i],cv2.COLOR_BGR2RGB)
  dsm = cv2.imread(dsm_dir[i],0)

  



  print("\n----- image num {}  ".format(i+1))
  print("----- input shape image: ", image.shape)
  #print("----- input label image: ", label.shape)
  #print("----- input dsm image: ", dsm.shape)


  for j in range(len(crop_size)):

    kernel_size, stride = crop_size[j], crop_size[j]

    extracted_image_pathces , out_image_size = extract_array_patches(image, kernel_size, stride)
    extracted_label_pathces , out_label_size = extract_array_patches(label, kernel_size, stride) 
    extracted_dsm_pathces , _ = extract_array_patches(dsm, kernel_size, stride) 

    print("\n----- input shape extracted_image_pathces & original image size: ", extracted_image_pathces.shape , out_image_size)
    #print("----- input shape extracted_label_pathces & original label size: ", extracted_label_pathces.shape , out_image_size) 
    #print("----- input shape extracted_label_pathces & original label size: ", extracted_dsm_pathces.shape , out_image_size) 

    B,C,W,H = extracted_image_pathces.shape

    for k in range(B):
      sample_image = extracted_image_pathces[k].numpy().transpose(1,2,0)
      sample_label = extracted_label_pathces[k].numpy().transpose(1,2,0)
      sample_dsm = extracted_dsm_pathces[k,0].numpy()

      cv2.imwrite('{}/{}.png'.format(path_img , counter), sample_image)
      cv2.imwrite('{}/{}.png'.format(path_lbl , counter), sample_label)
      cv2.imwrite('{}/{}.png'.format(path_dsm , counter), sample_dsm)


      counter += 1 

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic =True

random.seed(config['RandomSeed'])
np.random.seed(config['RandomSeed'])
torch.manual_seed(config['RandomSeed'])
torch.cuda.manual_seed(config['RandomSeed'])

    
source_dir_train = '/content/Train/'
if not os.path.isdir(source_dir_train):
    os.mkdir(source_dir_train)

path_img_train =  source_dir_train + "images"
if not os.path.isdir(path_img_train):
  os.mkdir(path_img_train)

path_lbl_train =  source_dir_train + "labels"
if not os.path.isdir(path_lbl_train):
  os.mkdir(path_lbl_train)

path_dsm_train =  source_dir_train + "dsms"
if not os.path.isdir(path_dsm_train):
  os.mkdir(path_dsm_train)


source_dir_valid = '/content/Valid/'
if not os.path.isdir(source_dir_valid):
    os.mkdir(source_dir_valid)

path_img_valid =  source_dir_valid + "images"
if not os.path.isdir(path_img_valid):
  os.mkdir(path_img_valid)

path_lbl_valid =  source_dir_valid + "labels"
if not os.path.isdir(path_lbl_valid):
  os.mkdir(path_lbl_valid)

path_dsm_valid =  source_dir_valid + "dsms"
if not os.path.isdir(path_dsm_valid):
  os.mkdir(path_dsm_valid)
    
file_names = os.listdir(path_img)

dataset_size = len(file_names)
indices = sorted(list(range(dataset_size)))
random.shuffle(indices)
split = int(np.floor(0.20 * dataset_size))

train_indices, val_indices = indices[split:], indices[:split]

train_file_names = []
valid_file_names = []


for i in train_indices:
  train_file_names.append(file_names[i])

  
for j in val_indices:
  valid_file_names.append(file_names[j])    



for file_name in train_file_names:
    shutil.move(os.path.join(path_img, file_name), path_img_train)
    shutil.move(os.path.join(path_lbl, file_name), path_lbl_train)
    shutil.move(os.path.join(path_dsm, file_name), path_dsm_train)


for file_name in valid_file_names:
    shutil.move(os.path.join(path_img, file_name), path_img_valid)
    shutil.move(os.path.join(path_lbl, file_name), path_lbl_valid)
    shutil.move(os.path.join(path_dsm, file_name), path_dsm_valid)

image_dir = sorted(glob.glob('/content/drive/MyDrive/Vahingen_Data/Dataset/Train/Image/*.tif'))
label_dir = sorted(glob.glob('/content/drive/MyDrive/Vahingen_Data/Dataset/Train/Label/*.tif'))
dsm_dir = sorted(glob.glob('/content/drive/MyDrive/Vahingen_Data/Dataset/Train/DSME/*.png'))

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic =True

random.seed(config['RandomSeed'])
np.random.seed(config['RandomSeed'])
torch.manual_seed(config['RandomSeed'])
torch.cuda.manual_seed(config['RandomSeed'])


train_path_img = glob.glob('/content/Train/images/*.png')
train_path_lbl = glob.glob('/content/Train/labels/*.png')
train_path_dsm = glob.glob('/content/Train/dsms/*.png')

valid_path_img = glob.glob('/content/Valid/images/*.png')
valid_path_lbl = glob.glob('/content/Valid/labels/*.png')
valid_path_dsm = glob.glob('/content/Valid/dsms/*.png')




dataset_train = TrainImageLoader(train_path_img,
                                 train_path_lbl,
                                 train_path_dsm,
                                 config['image_size'])
print('number of input train data -------> {}'.format(dataset_train.__len__()))

dataset_valid = ValidImageLoader(valid_path_img,
                                 valid_path_lbl,
                                 valid_path_dsm,
                                 config['image_size'])

print('number of input valid data -------> {}'.format(dataset_valid.__len__()))




train_loader = torch.utils.data.DataLoader(dataset_train, 
                                           batch_size = config['batch_size'],
                                           num_workers=4,
                                           pin_memory =False,
                                           shuffle = True,
                                           drop_last = True)



valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                           batch_size = config['batch_size'], 
                                           num_workers = 4,
                                           pin_memory =False,
                                           shuffle = False,
                                           drop_last = True)


print('number of train batch -------> {}'.format(len(train_loader)))
print('number of valid batch -------> {}'.format(len(valid_loader)))

# show sample
inputs, targets, dsms = next(iter(valid_loader))

#out_image
out_image = torchvision.utils.make_grid(inputs, padding=3)
out_image = out_image.cpu().numpy().transpose(1,2,0)

#out_label
out_labeln = targets.numpy()
out_label = np.zeros((out_labeln.shape[0] , 3 , out_labeln[1,:,:].shape[0] ,out_labeln[1,:,:].shape[1] ) , dtype=np.uint8)
for i in range(out_labeln.shape[0]):
    temp = convert_to_color(out_labeln[i,:,:])
    temp = temp.transpose(2,0,1)
    out_label[i,:,:,:] = temp
    
out_label = torch.from_numpy(out_label)
out_label = torchvision.utils.make_grid(out_label, padding=3)
out_label = out_label.cpu().numpy().transpose(1,2,0)

#out_dsm
out_dsm = torchvision.utils.make_grid(dsms, padding=3)
out_dsm = out_dsm.cpu().numpy().transpose(1,2,0)

fig, axs = plt.subplots(nrows=3, dpi=300 )
#fig.tight_layout(pad=60)

axs[0].imshow(out_image)
axs[0].axis('off')
axs[0].set_title('Original Image', size=3)

axs[1].imshow(out_label)
axs[1].axis('off')
axs[1].set_title('Original Label', size=3)

axs[2].imshow(out_dsm,'gray')
axs[2].axis('off')
axs[2].set_title('Original dsm', size=3)


#define models

model = CMGFNet(2,True)
model.to(device)

model_total_parameters = count_parameters(model)

config['just_record'] = {'Epoch': [], 'valid_loss':[], 'train_loss':[], 'ACC': [], 'mean_IoU': [],'IoU_Building': [],'Dice_mean': [], 'Dice_Building': [],
                             'Recall': [], 'Precision': [],  }

config['best_record'] = {'Best_Epoch': [0], 'valid_loss':[0], 'ACC': [0], 'mean_IoU': [0],'IoU_Building': [0],'Dice_mean': [0], 'Dice_Building': [0],
                             'Recall': [0], 'Precision': [0], }


optimization = optim.Adamax(model.parameters(), 
                         lr=config['learning_rate'],
                         weight_decay=0.00001)

writer = SummaryWriter(log_path)
curr_epoch = 1

train_start = time.time()

for epoch in range(curr_epoch, config['max_epoch'] + 1):

    train(train_loader, model, optimization, epoch, device, writer, config)
    validate(valid_loader, model, optimization, epoch, device , writer, config)


elapsed_time = time.time() - train_start

writer.add_text('train best quantitative results: ' + config['model_name'] , str(config['best_record']))


training_information = {'dataset_name': [config['dataset_name']],
                        'model_name': [config['model_name']],
                        'file_position':[colab_account_position],
                        'loss_function_criteria':loss_function_criteria,
                        'num_epoch': [config['max_epoch']],
                        'start_learning_rate': [config['learning_rate']],
                        'batch_size': [config['batch_size']],
                        'process_time': [elapsed_time] ,
                        'total_parameters': [model_total_parameters],
                        'description': description,}

information_all = pd.DataFrame.from_dict(training_information)
information_all.to_csv(log_path + '/information_' + config['model_name'] + '.csv')

data_frame_all = pd.DataFrame.from_dict(config['just_record'])
data_frame_all.to_csv(log_path + '/just_record_' + config['model_name'] + '.csv')

data_frame_best = pd.DataFrame.from_dict(config['best_record'])
data_frame_best.to_csv(log_path + '/best_record_' + config['model_name'] + '.csv')

writer.close()
