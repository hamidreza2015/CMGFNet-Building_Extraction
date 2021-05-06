#version 02
class Gated_Fusion(nn.Module):
    
    def __init__(self, in_channels):
        super().__init__()
        
        self.gate = nn.Sequential(            
            nn.Conv2d(2 * in_channels, in_channels,kernel_size=1, padding=0),
            nn.Sigmoid(),
            )
        
    def forward(self, x,y):
      out = torch.cat([x,y], dim=1)
      G = self.gate(out)
      
      PG = x * G
      FG = y * (1-G)

      
      return torch.cat([FG , PG], dim=1)
