class Discriminator(nn.Module):
    def __init__(self, n_class=21):
        super(Discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(n_class, 64, 3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 2, 3, stride=1, padding=1)

    def forward(self, x):
        res = F.relu(self.conv1_1(x),inplace=False)
        res = F.relu(self.conv1_2(res),inplace=False)
        res = F.max_pool2d(res, 2, stride=2)
        res = F.relu(self.conv2_1(res),inplace=False)
        res = F.max_pool2d(res, 2, stride=2)
        res = self.conv3_2(res)
        res = F.avg_pool2d(res, kernel_size=(res.shape[2], res.shape[3]))
        res = F.softmax(res)
        
        return res.view(8,-1).transpose(0,1)[0]


class Generator(nn.Module):
    
    def __init__(self, num_classes, pretrained=False):
        
        super().__init__()
        
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        # RGB Encoder Part
            
        self.resnet_features = torchvision.models.resnet18(pretrained=pretrained)
        
        self.enc_rgb1 = nn.Sequential(self.resnet_features.conv1,
                                    self.resnet_features.bn1,
                                    self.resnet_features.relu,)
        self.enc_rgb2 = nn.Sequential(self.resnet_features.maxpool,
                                    self.resnet_features.layer1)
        
        self.enc_rgb3 = self.resnet_features.layer2
        self.enc_rgb4 = self.resnet_features.layer3
        self.enc_rgb5 = self.resnet_features.layer4

        self.pool = nn.MaxPool2d(2)
        
        self.side6 = nn.Conv2d(512, 16, kernel_size=1, padding=0)
        self.side5 = nn.Conv2d(512, 16, kernel_size=1, padding=0)
        self.side4 = nn.Conv2d(256, 16, kernel_size=1, padding=0)
        self.side3 = nn.Conv2d(128, 16, kernel_size=1, padding=0)
        self.side2 = nn.Conv2d(64, 16, kernel_size=1, padding=0)
        self.side1 = nn.Conv2d(64, 16, kernel_size=1, padding=0)
        
                                 
        self.dconv6 = decoder_block(16   , 16)
        self.dconv5 = decoder_block(16 + 16  , 16)
        self.dconv4 = decoder_block(16 + 16  , 16)
        self.dconv3 = decoder_block(16 + 16  , 16)
        self.dconv2 = decoder_block(16 + 16  , 16)
        self.dconv1 = decoder_block(16 + 16  , 16)
        
        
        self.final = nn.Sequential(
            nn.Conv2d(16, self.num_classes, kernel_size=1, padding=0),
            nn.BatchNorm2d(self.num_classes),
            nn.ReLU(inplace=True),
            )
        
          
    def forward(self, x_rgb):
        

        # rgb_encoder
        
        x1 = self.enc_rgb1(x_rgb)         # bs * 64 * W/2 * H/2
        side1 = self.side1(x1)

        x2 = self.enc_rgb2(x1)            # bs * 64 * W/4 * H/4
        side2 = self.side2(x2)


        x3 = self.enc_rgb3(x2)            # bs * 128 * W/8 * H/8
        side3 = self.side3(x3)      

        
        x4 = self.enc_rgb4(x3)            # bs * 256 * W/16 * H/16  
        side4 = self.side4(x4)    
 
        
        x5 = self.enc_rgb5(x4)            # bs * 512 * W/32 * H/32
        side5 = self.side5(x5)


        x6 =  self.pool(x5)
        side6 = self.side6(x6)     

        out = self.dconv6(side6)         
         
        FG = torch.cat((side5,out),dim=1)      
        out = self.dconv5(FG)     
        
        FG = torch.cat((side4,out),dim=1)      
        out = self.dconv4(FG)  

        FG = torch.cat((side3,out),dim=1)    
        out = self.dconv3(FG)

        FG = torch.cat((side2,out),dim=1)    
        out = self.dconv2(FG)   

        FG = torch.cat((side1,out),dim=1)    
        out = self.dconv1(FG)   
                  
        out = self.final(out)
       
        return out
