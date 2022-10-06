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


class ValidImageLoader(Dataset):
    def __init__(
        self, 
        root, ):
        
        self.root = root        
        self.files = collections.defaultdict(list)
                
        file_list = os.listdir(self.root + '/images')
        self.files['valid'] = file_list
            
    def __len__(self):
        return len(self.files['valid'])
        
        
    def __getitem__(self,index):
        
        image_name = self.files['valid'][index]
        
        image_path = self.root + '/' + 'images' + '/' + image_name
                
        image = io.imread(image_path)
        image = torchvision.transforms.functional.to_tensor(image)
        
        label_path = image_path.replace('images','gt')
        label = convert_from_color(io.imread(label_path))
        label = torch.from_numpy(label).long()


        return image , label


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

def one_hot_encode(label, n_classes, requires_grad=True):
    """Return One Hot Label"""
    device = label.device
    one_hot_label = torch.eye(
        n_classes, device=device, requires_grad=requires_grad)[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label

def poly_lr_scheduler(optimizer,
                      init_lr,
                      iter,
                      lr_decay_iter=1,
                      max_iter=20000,
                      power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
    """
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

def train(train_loader, net, optimizer, epoch, device, writer,  config):

    net.train()

    train_loss = averageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    
    with tqdm(total=len(train_loader), file=sys.stdout , desc='train [epoch %d]'%epoch , unit='Batch_Size') as train_bar:
      
      for _, (image , label) in enumerate(train_loader):
        
        assert image.size()[2:] == label.size()[1:]

        N = image.size(0)
        image = image.to(device)
        label = label.to(device)

        optimizer = poly_lr_scheduler(optimizer ,
                                      init_lr=config['learning_rate'], 
                                      iter=curr_iter, 
                                      lr_decay_iter=1, 
                                      max_iter=((config['train_sample'] // config['batch_size']) * config['max_epoch']), 
                                      power=config['lr_power'])
        
        optimizer.zero_grad()
        
      
        outputs = net(image)
        
        loss = nn.BCEWithLogitsLoss()(outputs, one_hot_encode(label,2,requires_grad=False))
             

        loss.backward()
        
        optimizer.step()

        train_loss.update(loss.item() , N)

        curr_iter += 1
        writer.add_scalar('train_loss', train_loss.avg, curr_iter)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], curr_iter)

        train_bar.set_postfix(train_loss = train_loss.avg)
        train_bar.update()

    #print('train [epoch %d],  [train_loss  %.5f] '%(epoch ,train_loss.avg))
    writer.add_scalar('train_loss_per_epoch', train_loss.avg, epoch)  


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


def get_validation_metrics(groundtruth, predicted):
    validation_metrics = {}
    num_classes = 2
    cm_matrix = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predicted, groundtruth):
        cm_matrix += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
        
    overal_accuracy = np.diag(cm_matrix).sum() / cm_matrix.sum()
    
    iu = np.diag(cm_matrix) / (cm_matrix.sum(axis=1) + cm_matrix.sum(axis=0) - np.diag(cm_matrix))
    mean_iu = np.nanmean(iu)
    
    recall = np.diag(cm_matrix) / cm_matrix.sum(axis=0)
    mean_recall = np.nanmean(recall)
    
    precision = np.diag(cm_matrix) / cm_matrix.sum(axis=1)
    mean_precision = np.nanmean(precision)

    F1_Score = (2*precision*recall)/(precision + recall)
    F1_Score_mean = np.nanmean(F1_Score)

    validation_metrics['F1_Score'] = F1_Score_mean
    validation_metrics['MIoU'] = mean_iu
    validation_metrics['Accuracy'] = overal_accuracy
    validation_metrics['Precision'] = mean_precision
    validation_metrics['Recall'] = mean_recall
        
    return validation_metrics

def validate(val_loader, net, optimizer, epoch, device , writer, config, log):

    net.eval()
  
    val_loss_meter = averageMeter()
    gts_all, predictions_all = [], []
  
    with torch.no_grad():

        for  img_id, (img,gt_mask) in enumerate(val_loader):
          inputs, label = img.to(device) ,gt_mask.to(device)
          N = inputs.size(0)
      
          inputs = inputs.to(device)
          
      
          outputs = net(inputs)
          
          val_loss = nn.BCEWithLogitsLoss()(outputs, one_hot_encode(label,2,False))

          
          predictions = outputs.data.max(1)[1].cpu().numpy() 
      
          val_loss_meter.update(val_loss.item() , N)
          
          gts_all.append(label.data.cpu().numpy())
          predictions_all.append(predictions)

      
    metric = get_validation_metrics(gts_all, predictions_all)

 
    if metric['MIoU'] > config['best_record']['mean_iu']:
        config['best_record']['Val_Loss'] = val_loss_meter.avg
        config['best_record']['Epoch'] = epoch
        config['best_record']['ACC'] = metric['Accuracy']
        config['best_record']['mean_iu'] = metric['MIoU']
        config['best_record']['Recall'] = metric['Recall']
        config['best_record']['Precision'] = metric['Precision']
        config['best_record']['F1Score'] = metric['F1_Score']


    
               
        
        torch.save(net.state_dict(),'drive/My Drive/model_{}.pt'.format(config['model_name']))
        #torch.save(optimizer.state_dict(), 'drive/My Drive/optimizer_{}.pt'.format(config['model_name']))
        
    
    
    
    print_validation_report = '[epoch %d], [val loss %.5f], [ACC %.5f], [mean_iu %.5f], [Recall %.5f] ,[Precision %.5f] ,[F_measure %.5f] ,[lr %.5f]' % (
        epoch, 
        val_loss_meter.avg, 
        metric['Accuracy'],  
        metric['MIoU'], 
        metric['Recall'],
        metric['Precision'], 
        metric['F1_Score'],
        
        optimizer.param_groups[0]['lr'])
        
    
    print_best_record = 'best record: [epoch %d], [val loss %.5f], [ACC %.5f], [mean_iu %.5f], [Recall %.5f] ,[Precision %.5f] ,[F_measure %.5f] ' % (
        config['best_record']['Epoch'],
        config['best_record']['Val_Loss'], 
        config['best_record']['ACC'], 
        config['best_record']['mean_iu'],
        config['best_record']['Recall'], 
        config['best_record']['Precision'],
        config['best_record']['F1Score'],      
        )
        
    log.log("\n---------------------------------------------------------------------")

    log.log(print_validation_report)
    log.log(print_best_record)    

    log.log("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    

    writer.add_scalar('val_loss', val_loss_meter.avg, epoch)
    writer.add_scalar('Overall Accuracy', metric['Accuracy'], epoch)
    writer.add_scalar('Mean_iu', metric['MIoU'], epoch)
    writer.add_scalar('Recall', metric['Recall'], epoch)
    writer.add_scalar('Precision', metric['Precision'], epoch)
    writer.add_scalar('F_measure', metric['F1_Score'], epoch)