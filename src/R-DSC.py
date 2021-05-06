class Upsample(nn.Module):
    
    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode,align_corners=True)
        return x

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout,kernel_size=3, padding=1):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size = kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out    

class decoder_block(nn.Module):
    def __init__(self, 
                 input_channels, 
                 output_channels):
        
        super(decoder_block, self).__init__()
        
        self.identity = nn.Sequential(
            Upsample(2, mode="bilinear"),
            nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)
            )

        self.decode = nn.Sequential(
            Upsample(2, mode="bilinear"),
            nn.BatchNorm2d(input_channels),
            depthwise_separable_conv(input_channels,input_channels),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            depthwise_separable_conv(input_channels,output_channels),
            nn.BatchNorm2d(output_channels),
            )
        
   
    def forward(self,x):
      
      residual = self.identity(x)
      
      out = self.decode(x)

      out += residual

      return out
