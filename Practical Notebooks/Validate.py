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
