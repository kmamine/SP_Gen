import math
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F

#!pip install kornia
from itertools import product
from kornia.geometry import SpatialSoftArgmax2d
np.random.seed(128)
torch.manual_seed(128)

# Conv2D Layer with batch normalization and ReLU activation  
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

# 1x1 Conv2D Layer with batch normalization and ReLU activation 
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )



class ChannelGate(nn.Module):
    def __init__(self, gate_channels=20, reduction_ratio=4):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.bn2d = nn.BatchNorm2d(gate_channels)
        self.bn1d = nn.BatchNorm1d(gate_channels)
        self.mlp_avgp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels),
            nn.ReLU(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio,gate_channels // reduction_ratio),
            #nn.BatchNorm1d(gate_channels // nn.ReLU()eduction_ratio),
            nn.ReLU()
            )

        self.mlp_maxgp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels),
            nn.ReLU(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio,gate_channels // reduction_ratio),
            #nn.BatchNorm1d(num_features=gate_channels // reduction_ratio),
            nn.ReLU()
            )
        
        self.merge = nn.Sequential(
            Flatten(),
            nn.Linear(2*(gate_channels // reduction_ratio),gate_channels),
            #nn.BatchNorm1d(num_features=gate_channels),
            nn.Dropout(p=0.2),
            nn.Sigmoid()
        )

        for name, param in self.named_parameters():

            if "weight" in name and 'bn' not in name:
                    nn.init.xavier_normal(param)


    def forward(self, x):
       
        # Normalize 

        x= (x-x.min())/(x.max()-x.min()+0.0000001)
        x = self.bn2d(x)
        channel_att_sum = None
        
        # Calculate Global average pooling
        
        avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
     

        avgp_v = self.mlp_avgp( avg_pool )
        

        max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))

        maxgp_v = self.mlp_maxgp( max_pool )
 
        channel_selction = self.merge(torch.cat((avgp_v, maxgp_v), dim=1))
 

        return channel_selction

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


#  Invert Gradient sign Layer for unsupervised domain adaptation   
from torch.autograd import Function

class GradientReversalFn(Function):
  @staticmethod
  def forward(ctx, x, alpha=0.1):
    # Store context for backprop
    ctx.alpha = alpha
    
    # Forward pass is a no-op
    return x.view_as(x)

  @staticmethod
  def backward(ctx, grad_output):
    # Backward pass is just to -alpha the gradient
    output = grad_output.neg() * ctx.alpha

    # Must return same number as inputs to forward()
    return output, None

RevGrad = GradientReversalFn.apply

#  mobilebet modules
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, omit_stride=False,
                 no_res_connect=False, dropout=0., bn_momentum=0.1,
                 batchnorm=None):
        super().__init__()
        self.out_channels = oup
        self.stride = stride
        self.omit_stride = omit_stride
        self.use_res_connect = not no_res_connect and\
            self.stride == 1 and inp == oup
        self.dropout = dropout
        actual_stride = self.stride if not self.omit_stride else 1
        if batchnorm is None:
            def batchnorm(num_features):
                return nn.BatchNorm2d(num_features, momentum=bn_momentum)

        assert actual_stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        if expand_ratio == 1:
            modules = [
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, actual_stride, 1,
                          groups=hidden_dim, bias=False),
                batchnorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                batchnorm(oup),
            ]
            if self.dropout > 0:
                modules.append(nn.Dropout2d(self.dropout))
            self.conv = nn.Sequential(*modules)
        else:
            modules = [
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                batchnorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, actual_stride, 1,
                          groups=hidden_dim, bias=False),
                batchnorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                batchnorm(oup),
            ]
            if self.dropout > 0:
                modules.insert(3, nn.Dropout2d(self.dropout))
            self.conv = nn.Sequential(*modules)
            self._initialize_weights()

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



# MobileNet backbone with custom modification for Unsupervised DA 
class MobileNetV2(nn.Module):
    def __init__(self, widen_factor=1., pretrained=True,
                 last_channel=1280, input_channel=32):
        super().__init__()
        self.widen_factor = widen_factor
        self.pretrained = pretrained
        self.last_channel = last_channel
        self.input_channel = input_channel

        block = InvertedResidual
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(self.input_channel * widen_factor)
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * widen_factor)
            for i in range(n):
                if i == 0:
                    self.features.append(block(
                        input_channel, output_channel, s, expand_ratio=t,
                        omit_stride=True))
                else:
                    self.features.append(block(
                        input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layerself.BCEs
        if self.last_channel is not None:
            output_channel = int(self.last_channel * widen_factor)\
                if widen_factor > 1.0 else self.last_channel
            self.features.append(conv_1x1_bn(input_channel, output_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        self.out_channels = output_channel
        self.feat_1x_channels = int(
            interverted_residual_setting[-1][1] * widen_factor)
        self.feat_2x_channels = int(
            interverted_residual_setting[-2][1] * widen_factor)
        self.feat_4x_channels = int(
            interverted_residual_setting[-4][1] * widen_factor)
        self.feat_8x_channels = int(
            interverted_residual_setting[-5][1] * widen_factor)

        if self.pretrained:
            self._initialize_weights()

        else:
            self._initialize_weights()

    def forward(self, x):
        feat_2x, feat_4x, feat_8x = None, None, None
        for idx, module in enumerate(self.features._modules.values()):
            
            x = module(x)
            if idx == 4 :
                feat_8x = x.clone()
            elif idx == 7:
                feat_4x = x.clone()
            elif idx == 14:
                feat_2x = x.clone()
            if idx > 0 and hasattr(module, 'stride') and module.stride != 1:   
                x = x[..., ::2, ::2]
            
        # Return multiple layer in order to apply Hierarchical Domain-Adapted Feature Learning 'https://arxiv.org/abs/2010.01220'
        # Improvment ..!!! DOI : 10.1109/CVPR42600.2020.00410
        
        return x, feat_2x, feat_4x#, feat_8x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            
#k_folds
import torchvision.models as models
# Final scanpath model 
class ScanPathModel(nn.Module):
    """
    
    """
    def  __init__(self):
          super(ScanPathModel,self).__init__()

          # inti the encoder as standerd mobilenet
          #self.encoder = MobileNetV2()
          mobilenet_v2 = models.mobilenet_v2(pretrained=True)
          self.encoder = mobilenet_v2.features
          # set the decoder
          self.scanpath_decoder = torch.nn.Sequential(*[
            nn.Conv2d(1296, 648, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(648, 648, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(648, 324, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(324, 324, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(324, 162, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(162, 162, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(162, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(20, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
         
          ]) 
          self.soft_sam  = SpatialSoftArgmax2d(normalized_coordinates=False) 
          
          self.rand_mul =  nn.Parameter(torch.rand(1))
          # Set the Hierarchical Domain-Adapted GRL!
          self.domain = False
          self.domain_classifier =  torch.nn.Sequential(nn.Linear(1280, 360), nn.ReLU(), nn.Linear(360, 2), nn.LogSoftmax(dim=1))

          # Set the learnable Prior Maps! 
          self.n_gaussians = 16
          self.set_gaussians()

          #set the chanflatten(x, 1)el selection net
          self.ChannelGate = ChannelGate(gate_channels=20)
          for name, param in self.scanpath_decoder.named_parameters():
                if "weight" in name :
                    nn.init.xavier_normal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)
    
          for param in self.encoder.parameters():
            param.requires_grad = False

        #   for param in self.gaussians.parameters():
          self.gaussians.requires_grad = False


    def _initialize_gaussians(self, n_gaussians):
        """
        Return initialized Gaussian parameters.
        
        """
        gaussians = torch.Tensor([
                    list(product([0.25, 0.5, 0.75], repeat=2)) +
                    [(0.5, 0.25), (0.5, 0.5), (0.5, 0.75)] +
                    [(0.25, 0.5), (0.5, 0.5), (0.75, 0.5)] +
                    [(0.5, 0.5)],
                    [(-1.5, -1.5)] * 9 + [(0, -1.5)] * 3 + [(-1.5, 0)] * 3 +
                    [(0, 0)],
            ]).permute(1, 2, 0)
        gaussians = nn.Parameter(gaussians, requires_grad=True)
        return gaussians

    def set_gaussians(self):
        """Set Gaussian parameters."""      
        self.__setattr__(
            'gaussians',
            self._initialize_gaussians(self.n_gaussians))
    @staticmethod
    def _make_gaussian_maps(x, gaussians, size=None, scaling=6.):
              """Construct prior maps from Gaussian parameters."""
              if size is None:
                  size = x.shape[-2:]
                  bs = x.shape[0]
              else:
                  size = [size] * 2
                  bs = 1
              dtype = x.dtype
              device = x.device

              gaussian_maps = []
              map_template = torch.ones(*size, dtype=dtype, device=device)
              meshgrids = torch.meshgrid(
                  [torch.linspace(0, 1, size[0], dtype=dtype, device=device),
                  torch.linspace(0, 1, size[1], dtype=dtype, device=device),])

              for gaussian_idx, yx_mu_logstd in enumerate(torch.unbind(gaussians)):
                  map = map_template.clone()
                  for mu_logstd, mgrid in zip(yx_mu_logstd, meshgrids):
                      mu = mu_logstd[0]
                      std = torch.exp(mu_logstd[1])
                      map *= torch.exp(-((mgrid - mu) / std) ** 2 / 2)
                  map *= scaling
                  gaussian_maps.append(map)
              gaussian_maps = torch.stack(gaussian_maps)
              gaussian_maps = gaussian_maps.unsqueeze(0).expand(bs, -1, -1, -1)
              return gaussian_maps



    def _get_gaussian_maps(self, x, **kwargs):
              """Return the constructed Gaussian prior maps."""
              
              gaussians = self.__getattr__("gaussians")
              gaussian_maps = self._make_gaussian_maps(x, gaussians, **kwargs)
              return gaussian_maps

    def forward(self,input,temp = 0.5):
      
              #bottel, mid_1, mid_2 = self.encoder(input)
              bottel = self.encoder(input)
              adapt = torch.nn.AdaptiveAvgPool2d((1,1))
              gaussian_maps = self._get_gaussian_maps(bottel)
              
              # Shuffle Guasia maps
              #print(gaussian_maps[0,0,:,:])
              #pos = list(range(gaussian_maps.shape[1]))
              #np.random.shuffle(pos)
              
              #print(pos)
              #gaussian_maps = gaussian_maps[:,pos,:,:]
              #print(gaussian_maps[0,0,:,:])
              #gaussian_maps = gaussian_maps * 2 

              bottel_neck = torch.cat((bottel, gaussian_maps), dim=1)
              out = self.scanpath_decoder(bottel_neck)
             
              
            #   rand_noise = torch.rand(*(out.shape)).to(self.rand_mul.device)  #  Uniform  noise
              rand_noise = torch.randn(*(out.shape)).to(self.rand_mul.device)  #  Gaussian  noise
              #print(rand.shape, out.shape)
              mean_noise = torch.mean(rand_noise, 1 )# torch.nn.AdaptiveAvgPool2d((1,1))
              #print(mean_noise.shape, rand_noise.shape)
              
              #print(rand_noise.device , self.rand_mul.device)
              # o =   self.rand_mul.reshape((-1,1,1,1)) * rand_noise
              #   out = out + self.rand_mul.reshape((-1,1,1,1)) * rand_noise
              out = (1-temp)  * out +  temp * rand_noise

              mask_vector = self.ChannelGate(out)
              mask_ =mask_vector
              #mask_vector = mask_vector> self.treshold
              mask_vector = mask_vector>mask_vector.mean()
              #mask_vector = mask_*mask_vector + (1- mask_.detach()*mask_vector)*mask_vector
             
              mask_ = mask_.view(-1,20,1,1)

              result = self.soft_sam(out*mask_)/torch.tensor(out.shape[-2:]).float().cuda()

              if self.domain:   
                    reverse_features = RevGrad(Flatten()(adapt(bottel)), 1.0)
                    domain_out = self.domain_classifier(reverse_features)
                    return result ,  gaussian_maps, domain_out, mask_vector   
                
              return result, gaussian_maps, mask_vector     
            

if __name__ == '__main__' : 
    with torch.no_grad():
        model = ScanPathModel()
        model.cuda()
        #model.load_state_dict(torch.load('./weight/icme_  0.pt'), strict = True)
        #ut, gaussian_maps, domain_out,mask_vector  = model(torch.rand([1,3,192, 256]).cuda())
        #domain_creterion = torch.nn.NLLLoss()
        
        #model.load_state_dict(torch.load('./weight_fs/icme_ 88.pt'), strict = False)


        ut1, _, _  = model(torch.rand([1,3,192, 256]).cuda())
        ut2, _, _  = model(torch.rand([1,3,192, 256]).cuda())

    print(ut1)
    print(ut2)

"""
def selct_chanels(output, mask): 
    bs,chanel,cord = output.shape 
    repeted_mask = mask_vector.int().view(-1,chanel,1).repeat(1,1,cord)
    result = output*repeted_mask
    return result[mask],mask.sum(1)

class LOSS(nn.Module):
    def __init__(self):
        super(LOSS, self).__init__()
        self.BCE = nn.BCELoss()    
        self.MSE = nn.MSELoss()
    def forward(self, predict,mask_vector, scanpath):
        predict, lenght = selct_chanels(predict,mask_vector)
        return self.BCE(predict,scanpath) + 0.0001*torch.sqrt(self.MSE(lenght,scanpath.shape[1]))

loss = LOSS()

loss(out,mask_vector,out)
"""
"""
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
def show_tensor_images(image_tensor, num_images=16, size=(1, 6, 8)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    print(image_unflat.shape)
    image_grid = make_grid(image_unflat, nrow=4)
    print(image_grid[0].shape)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

 
show_tensor_images(priors)
"""


