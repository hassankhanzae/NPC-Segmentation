import torch

class DoubleConv(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        self.step = torch.nn.Sequential(torch.nn.Conv3d(in_channels, out_channels, 3, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv3d(out_channels, out_channels, 3, padding=1),
                                        torch.nn.ReLU())
        
    def forward(self, X):
        return self.step(X)

class TripleConv(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        self.step = torch.nn.Sequential(torch.nn.Conv3d(in_channels, out_channels, 1),
                                        torch.nn.BatchNorm3d(out_channels),
                                         torch.nn.ReLU(),
                                        torch.nn.Conv3d(out_channels, out_channels, 3, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv3d(out_channels, out_channels, 3, padding=1),
                                        torch.nn.ReLU())
        
    def forward(self, X):
        return self.step(X)    
    
class up_conv(nn.Module):
    def __init__(self,in_channels,out_channels):
        
        super().__init__()
        self.up = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2,  mode="trilinear"),
            torch.nn.Conv3d(in_channels,out_channels,3,stride=1,padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU()
        )

    def forward(self,x):
        x = self.up(x)
        return x
    
    
class Recurrent_block(nn.Module):
    def __init__(self, out_channels,t=2):
        
        super().__init__()
        self.t = t
        self.conv = nn.Sequential(
            torch.nn.Conv3d(out_channels,out_channels,3,stride=1,padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU()
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
    
    
class RRCNN_block(nn.Module):
    def __init__(self,in_channels,out_channels,t=2):
        super().__init__()
        
        self.RCNN = torch.nn.Sequential(
            Recurrent_block(out_channels,t=2),
            Recurrent_block(out_channels,t=2)
        )
        self.Conv_1x1 = torch.nn.Conv3d(in_channels,out_channels,1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1

class BtResidualBlock(nn.Module):
    def __init__(self, self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        
        self.conv1 = torch.nn.Sequential(torch.nn.Conv3d(in_channels, out_channels, 1,padding=0),
                                        torch.nn.ReLU())

       
        self.conv2 = torch.nn.Sequential(torch.nn.Conv3d(in_channels, out_channels, 3,padding=0),
                                        torch.nn.ReLU())

        
        self.conv3 = torch.nn.Sequential(torch.nn.Conv3d(in_channels, out_channels, 1,padding=0),
                                        torch.nn.ReLU())

    def forward(self, x):

        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        x_out = self.conv3(x_out)

        return x + x_out
    
    

class U_Net(torch.nn.Module):
    

    def __init__(self):
       
        super().__init__()
        
        
        self.layer1 = DoubleConv(1, 64)
        self.layer2 = DoubleConv(64, 128)
        self.layer3 = DoubleConv(128, 256)
        self.layer4 = DoubleConv(256, 512)
        self.layer5 = DoubleConv(512, 1024)


        
        self.layer6 = DoubleConv(1024 + 512, 512)
        self.layer7 = DoubleConv(512 + 256, 256)
        self.layer8 = DoubleConv(256 + 128, 128)
        self.layer9 = DoubleConv(128 + 64, 64)
        self.layer10 = torch.nn.Conv3d(64, 3, 1)  # background, Nasphyrangeal, NPC tumor
      

        self.maxpool = torch.nn.MaxPool3d(2)

    def forward(self, x):
        
       
        x1 = self.layer1(x)
        x1m = self.maxpool(x1)
        
        
              
        x2 = self.layer2(x1m)
        x2m = self.maxpool(x2)
       

               
        x3 = self.layer3(x2m)
        x3m = self.maxpool(x3)
      
        
      
        x4 = self.layer4(x3m)
        x4m = self.maxpool(x4)
        
        x5 = self.layer5(x4m)
            
        x6 = torch.nn.Upsample(scale_factor=2, mode="trilinear")(x5)  
        x6 = torch.cat([x6, x4], dim=1)  # Skip-Connection
        x6 = self.layer6(x6)
              
        x7 = torch.nn.Upsample(scale_factor=2, mode="trilinear")(x6)        
        x7 = torch.cat([x7, x3], dim=1)  
        x7 = self.layer7(x7)
               
        x8 = torch.nn.Upsample(scale_factor=2, mode="trilinear")(x7)
        x8 = torch.cat([x8, x2], dim=1)       
        x8 = self.layer8(x8)
        
        x9 = torch.nn.Upsample(scale_factor=2, mode="trilinear")(x8)
        x9 = torch.cat([x9, x1], dim=1)       
        x9 = self.layer9(x9)
                
        ret = self.layer10(x9)
        return ret

    
class R2U_Net(nn.Module):
    def __init__(self, t=2):
        super().__init__()
        
        self.Maxpool = torch.nn.MaxPool3d(2)
        self.Upsample = nn.Upsample(scale_factor=2 , mode="trilinear")

        self.RRCNN1 = RRCNN_block(1,64,t=t)

        self.RRCNN2 = RRCNN_block(64,128,t=t)
        
        self.RRCNN3 = RRCNN_block(128,256,t=t)
        
        self.RRCNN4 = RRCNN_block(256,512,t=2)
        
        self.RRCNN5 = RRCNN_block(512,1024,t=2)
        

        self.Up5 = up_conv(1024,512)
        self.Up_RRCNN5 = RRCNN_block(1024,512,t=2)
        
        self.Up4 = up_conv(512,256)
        self.Up_RRCNN4 = RRCNN_block(512,256,t=2)
        
        self.Up3 = up_conv(256,128)
        self.Up_RRCNN3 = RRCNN_block(256,128,t=2)
        
        self.Up2 = up_conv(128,64)
        self.Up_RRCNN2 = RRCNN_block(128,64,t=2)

        self.Conv_1x1 = torch.nn.Conv3d(64,3,1)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class MU_Net(torch.nn.Module):
    

    def __init__(self):
       
        super().__init__()
        
        
        self.layer1 = TripleConv(1, 32)
        self.layer2 = TripleConv(32, 64)
        self.layer3 = TripleConv(64, 128)
        self.layer4 = TripleConv(128, 256)


        
        self.layer5 = BtResidualBlock(256 + 128, 128)
        self.layer6 = BtResidualBlock(128 + 64, 64)
        self.layer7 = BtResidualBlock(64 + 32, 32)
        self.layer8 = torch.nn.Conv3d(32, 3, 1)  # background, Nasphyrangeal, NPC tumor
      

        self.maxpool = torch.nn.MaxPool3d(2)

    def forward(self, x):
        
       
        x1 = self.layer1(x)
        x1m = self.maxpool(x1)
        
        
              
        x2 = self.layer2(x1m)
        x2m = self.maxpool(x2)
       

               
        x3 = self.layer3(x2m)
        x3m = self.maxpool(x3)
      
        
      
        x4 = self.layer4(x3m)
            
        x5 = torch.nn.Upsample(scale_factor=2, mode="trilinear")(x4)  
        x5 = torch.cat([x5, x3], dim=1)  # Skip-Connection
        x5 = self.layer5(x5)
              
        x6 = torch.nn.Upsample(scale_factor=2, mode="trilinear")(x5)        
        x6 = torch.cat([x6, x2], dim=1)  
        x6 = self.layer6(x6)
               
        x7 = torch.nn.Upsample(scale_factor=2, mode="trilinear")(x6)
        x7 = torch.cat([x7, x1], dim=1)       
        x7 = self.layer7(x7)
                
        ret = self.layer8(x7)
        return ret

    