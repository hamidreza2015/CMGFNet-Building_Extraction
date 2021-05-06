def validate(val_loader, net, optimizer, epoch, device , writer, config):

    net.eval()
    valid_loss = averageMeter()
  
    target_all, pred_all = [], []

    with torch.no_grad():

      for  _, (img, target, dsm) in enumerate(val_loader):

        img, target, dsm = img.to(device), target.to(device), dsm.to(device)
        N = img.size(0)

        outputs1,outputs2,outputs3 = net(img,dsm)
       
        loss_valid1 = nn.BCEWithLogitsLoss()(outputs1, class2one_hot(target,2).float()) 
        loss_valid2 = nn.BCEWithLogitsLoss()(outputs2, class2one_hot(target,2).float()) 
        loss_valid3 = nn.BCEWithLogitsLoss()(outputs3, class2one_hot(target,2).float()) 

        loss_valid = (loss_valid1 + loss_valid2 +loss_valid3)/3

        valid_loss.update(loss_valid.item() , N)

        outputs = F.softmax(outputs1, dim=1)
                
        
        prediction = outputs.data.max(1)[1].cpu().numpy() 

        target_all.append(target.data.cpu().numpy())
        pred_all.append(prediction)

        

    metric = get_validation_metrics(target_all, pred_all)


      
    config['just_record']['Epoch'].append(epoch)
    config['just_record']['valid_loss'].append(valid_loss.avg)
    config['just_record']['ACC'].append(metric['Accuracy'])
    config['just_record']['mean_IoU'].append(metric['MIoU'])
    config['just_record']['IoU_Building'].append(metric['IoU_Building'])
    config['just_record']['Recall'].append(metric['Recall'])
    config['just_record']['Precision'].append(metric['Precision'])
    config['just_record']['Dice_mean'].append(metric['Dice_mean'])
    config['just_record']['Dice_Building'].append(metric['Dice_Building'])

 
    if metric['MIoU'] > config['best_record']['mean_IoU'][-1] or epoch==1:
        config['best_record']['Best_Epoch'].append(epoch)
        config['best_record']['valid_loss'].append(valid_loss.avg)
        config['best_record']['ACC'].append(metric['Accuracy'])
        config['best_record']['mean_IoU'].append(metric['MIoU'])
        config['best_record']['IoU_Building'].append(metric['IoU_Building'])
        config['best_record']['Recall'].append(metric['Recall'])
        config['best_record']['Precision'].append(metric['Precision'])
        config['best_record']['Dice_mean'].append(metric['Dice_mean'])
        config['best_record']['Dice_Building'].append(metric['Dice_Building'])


        
        torch.save(net.state_dict(), log_path + '/' + 'model_{}.pt'.format(config['model_name']))
        torch.save(optimizer.state_dict(), log_path + '/' + 'optimizer_{}.pt'.format(config['model_name']))
        
    
    print_validation_report = '[epoch %d], [valid_loss %.4f], [ACC %.4f], [mean_iou %.4f], [Dice %.4f], [Recall %.4f] ,[Precision %.4f] ,[lr %.4f]' % (
        epoch, 
        valid_loss.avg,
        metric['Accuracy'],  
        metric['MIoU'], 
        metric['Dice_mean'],
        metric['Recall'],
        metric['Precision'], 
        
        
        optimizer.param_groups[0]['lr'])
        
    
    print_best_record = 'best record: [epoch %d],\n           [valid_loss %.4f], [ACC %.4f], [mean_iou %.4f], [Dice %.4f], [Recall %.4f] ,[Precision %.4f]  ' % (
        config['best_record']['Best_Epoch'][-1],
        config['best_record']['valid_loss'][-1],
        config['best_record']['ACC'][-1], 
        config['best_record']['mean_IoU'][-1],
        config['best_record']['Dice_mean'][-1], 
        config['best_record']['Recall'][-1], 
        config['best_record']['Precision'][-1],
          )
        
    print("--------------------------------------------------------------------------------------------")

    print(print_validation_report,'\n')
    print(print_best_record)  

    print("____________________________________________________________________________________________")
    print("____________________________________________________________________________________________\n\n")
    
    writer.add_scalar('valid_loss', valid_loss.avg, epoch)
    writer.add_scalar('Overall Accuracy', metric['Accuracy'], epoch)
    writer.add_scalar('Mean_iou', metric['MIoU'], epoch)
    writer.add_scalar('iou_building', metric['IoU_Building'], epoch)
    writer.add_scalar('Recall', metric['Recall'], epoch)
    writer.add_scalar('Precision', metric['Precision'], epoch)
    writer.add_scalar('Dice', metric['Dice_mean'], epoch)
    writer.add_scalar('Dice_Building', metric['Dice_Building'], epoch)
