def train(train_loader, net, optimizer, epoch, device, writer,  config ):

    net.train()

    train_loss = averageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    

    with tqdm(total=len(train_loader), file=sys.stdout , desc='train [epoch %d]'%epoch , unit='Batch_Size') as train_bar:
      
      for _, (image , label, dsm) in enumerate(train_loader):
        
        assert image.size()[2:] == label.size()[1:]

        N = image.size(0)
        image = image.to(device)
        label = label.to(device)
        dsm = dsm.to(device)

                
        optimizer.zero_grad()

        optimizer = poly_lr_scheduler(optimizer ,
                                      init_lr=config['learning_rate'], 
                                      iter=curr_iter, 
                                      lr_decay_iter=1, 
                                      max_iter=(len(train_loader) * config['max_epoch']), 
                                      power=config['lr_power'])
        
      
        outputs1,outputs2,outputs3 = net(image,dsm)
       
        loss1 = nn.BCEWithLogitsLoss()(outputs1, class2one_hot(label,2).float()) 
        loss2 = nn.BCEWithLogitsLoss()(outputs2, class2one_hot(label,2).float()) 
        loss3 = nn.BCEWithLogitsLoss()(outputs3, class2one_hot(label,2).float()) 

        loss = (loss1 + loss2 + loss3)/3

                        
        loss.backward()
        
        optimizer.step()
        
        
        train_loss.update(loss.item() , N)
        

        curr_iter += 1

        writer.add_scalar('train_loss', train_loss.avg, curr_iter)       
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], curr_iter)

        train_bar.set_postfix(train_loss = train_loss.avg)
        train_bar.update()

    
    
    writer.add_scalar('train_loss_per_epoch', train_loss.avg, epoch) 
    config['just_record']['train_loss'].append(train_loss.avg)
