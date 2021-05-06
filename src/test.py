
image_dir = sorted(glob.glob('/content/drive/MyDrive/Vahingen_Data/Dataset/Test/Image/*.tif'))
label_dir = sorted(glob.glob('/content/drive/MyDrive/Vahingen_Data/Dataset/Test/Label/*.tif'))
dsm_dir = sorted(glob.glob('/content/drive/MyDrive/Vahingen_Data/Dataset/Test/DSME/*.png'))

kernel_size =256 #@param {type: "number"}
stride =  128#@param {type: "number"}

method = '_stride{}'.format(stride)

test_name = model_name + method

# Original path for Result
test_path =  log_path + '/' + test_name + '/'
if not os.path.isdir(test_path):
    os.mkdir(test_path)


path_test_fp_tn_total = test_path + '/fp_tn'
if not os.path.isdir(path_test_fp_tn_total):
  os.mkdir(path_test_fp_tn_total)

path_test_pred_total = test_path + '/pred'
if not os.path.isdir(path_test_pred_total):
  os.mkdir(path_test_pred_total)

path_test_heat_total = test_path + '/Heat_Map'
if not os.path.isdir(path_test_heat_total):
  os.mkdir(path_test_heat_total)
  
  
  
Accuracy = []
Name = []
Dice_mean = []
Dice_Building = []
MIoU = []
IoU_Building = []
Precision = []
Recall = []


for i in range(len(label_dir)):
#for i in [5]:

  name = 'image_{}'.format(i+1)


  original_label_from_patches, original_pred_binary_map, original_pred_soft = make_result(image_dir[i],
                                                                                          label_dir[i],  
                                                                                          dsm_dir[i],                                                                                        
                                                                                          kernel_size , 
                                                                                          stride , 
                                                                                          model)
  
  output_fp_fn = plot_fp_fn(original_pred_binary_map.cpu().numpy().astype(np.int) ,original_label_from_patches[:,0,...].cpu().numpy())

  target_all = original_label_from_patches[0,0].cpu().numpy()
  predict_all = original_pred_binary_map[0].cpu().numpy()
  predsoft_all = original_pred_soft[0,0].cpu().numpy()

  save_pred_map(predict_all,
                path_test_pred_total,
                name, 
                model_name,
                mode='gray')
  
  save_pred_map(predsoft_all,
                path_test_heat_total,
                name, 
                model_name,
                mode='jet')
  
  save_pred_map(output_fp_fn[0].transpose(1,2,0),
                path_test_fp_tn_total,
                name, 
                model_name,
                mode=None)

  metric = get_validation_metrics(target_all.astype(np.int),predict_all.astype(np.int))  
  
  Name.append(name)
  Accuracy.append(metric['Accuracy'])
  Dice_mean.append(metric['Dice_mean'])
  Dice_Building.append(metric['Dice_Building'])
  MIoU.append(metric['MIoU'])
  IoU_Building.append(metric['IoU_Building'])
  Precision.append(metric['Precision'])
  Recall.append(metric['Recall'])

  
  print('--------> image {} Done!'.format(i+1))


result = PrettyTable()
result.field_names = ["im_name", "Accuracy", "Dice_mean","Dice_Building", "MIoU","IoU_Building","Precision","Recall"]

for j in range(len(Name)):

  result.add_row([Name[j], Accuracy[j], Dice_mean[j], Dice_Building[j], MIoU[j], IoU_Building[j], Precision[j], Recall[j]])

Name.append('Avg')
Accuracy.append(np.mean(Accuracy))
Dice_mean.append(np.mean(Dice_mean))
Dice_Building.append(np.mean(Dice_Building))
MIoU.append(np.mean(MIoU))
IoU_Building.append(np.mean(IoU_Building))
Precision.append(np.mean(Precision))
Recall.append(np.mean(Recall))

result.add_row([Name[len(Name)-1], Accuracy[len(Name)-1], Dice_mean[len(Name)-1], 
                Dice_Building[len(Name)-1], MIoU[len(Name)-1], 
                IoU_Building[len(Name)-1], Precision[len(Name)-1], Recall[len(Name)-1]])

print(result)
ptable_to_csv(result, test_path +'/'+'result'+'_'+ model_name + '.csv')
