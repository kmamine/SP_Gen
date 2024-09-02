import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Function

#!pip install kornia
from kornia.geometry import SpatialSoftArgmax2d

import numpy as np
import math
from itertools import product


# Set the random seed for reproducibility
np.random.seed(128)
torch.manual_seed(128)

# Conv2D Layer with batch normalization and ReLU activation  
def conv_bn(inp, oup, stride):
    """
    Conv2D Layer with batch normalization and ReLU activation
    Args:
    - inp: input channels
    - oup: output channels
    - stride: stride of the convolutional layer
    output:
    - nn.Sequential: Conv2D Layer with batch normalization and ReLU activation    
    """
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

# 1x1 Conv2D Layer with batch normalization and ReLU activation 
def conv_1x1_bn(inp, oup):
    """
    1x1 Conv2D Layer with batch normalization and ReLU activation
    Args:
    - inp: input channels
    - oup: output channels
    output:
    - nn.Sequential: 1x1 Conv2D Layer with batch normalization and ReLU activation
    """
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )



class ChannelGate(nn.Module):
    """
    ChannelGate: Channel Selection Module
    Args:
    - gate_channels: number of input channels
    - reduction_ratio: reduction ratio for the channel selection
    output:
    - channel_selction: Channel selection mask
    """
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
        """
        Forward pass of the Channel Selection Module
        Args:
        - x: input tensor
        output:
        - channel_selction: Channel selection mask
        """
       

        # Normalize 
        x= (x-x.min())/(x.max()-x.min()+0.0000001)
        x = self.bn2d(x)
        channel_att_sum = None
        
        # Calculate Global average pooling
        avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        avgp_v = self.mlp_avgp( avg_pool )
        
        # Calculate Global max pooling
        max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        maxgp_v = self.mlp_maxgp( max_pool )

        # Merge the two pooling results
        channel_selction = self.merge(torch.cat((avgp_v, maxgp_v), dim=1))
 

        return channel_selction

class Flatten(nn.Module):
    """
    Flatten: Flatten Layer
    Args:
    - x: input tensor
    output:
    - x: Flattened tensor
    """
    def forward(self, x):
        return x.view(x.size(0), -1)


#  Invert Gradient sign Layer for unsupervised domain adaptation   

class GradientReversalFn(Function):
  """
  GradientReversalFn: Invert Gradient sign Layer for unsupervised domain adaptation
  Args:
    - ctx: context
    - x: input tensor
    - alpha: alpha value for the gradient reversal
    output:
    - output: Inverted gradient

  """
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

#  Invert Gradient sign Layer for unsupervised domain adaptation
RevGrad = GradientReversalFn.apply

            
# Final scanpath model 
class ScanPathModel(nn.Module):
    """
    ScanPathModel: Scanpath Prediction Model with Hierarchical Domain-Adapted Gradient Reversal Layer
    Args:
    - domain: Domain adaptation flag
    output:
    - result: Scanpath prediction
    - gaussian_maps: Gaussian prior maps
    - domain_out: Domain classification output
    - mask_vector: Channel selection mask

    """
    def  __init__(self,domain=False):
          super(ScanPathModel,self).__init__()

          # inti the encoder as standerd mobilenet
          self.encoder = models.mobilenet_v2(pretrained=True).features
          
          
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
          
          # Set the Spatial Soft Argmax Layer
          self.soft_sam  = SpatialSoftArgmax2d(normalized_coordinates=False) 
          
          self.rand_mul =  nn.Parameter(torch.rand(1))
          
          
          # Set the Hierarchical Domain-Adapted Gradient Reversal Layer!
          self.domain = domain
          self.domain_classifier =  torch.nn.Sequential(nn.Linear(1280, 360), nn.ReLU(), nn.Linear(360, 2), nn.LogSoftmax(dim=1))

          # Set the learnable Prior Maps
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
        Return initialized Gaussian parameters, for the Gaussian Prior Maps.
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
              """
              Args:
                - input: input tensor (image)
                - temp: temperature for the spatial generative ability
                output:
                - result: Scanpath prediction
                - gaussian_maps: Gaussian prior maps
                - mask_vector: Channel selection mask
                - domain_out: Domain classification output (if domain adaptation is enabled)
                
              """
              bottel = self.encoder(input)
              adapt = torch.nn.AdaptiveAvgPool2d((1,1))
              gaussian_maps = self._get_gaussian_maps(bottel)

              bottel_neck = torch.cat((bottel, gaussian_maps), dim=1)
              out = self.scanpath_decoder(bottel_neck)
             
              # Add the spatial generative ability to the model through the random noise and temperature

            #   rand_noise = torch.rand(*(out.shape)).to(self.rand_mul.device)  #  Uniform  noise
              rand_noise = torch.randn(*(out.shape)).to(self.rand_mul.device)  #  Gaussian  noise
              mean_noise = torch.mean(rand_noise, 1 )# torch.nn.AdaptiveAvgPool2d((1,1))
              out = (1-temp)  * out +  temp * rand_noise


              # Adding the channel wise (number of predited fixation points)  generation ability to the model
              mask_vector = self.ChannelGate(out)
              mask_ =mask_vector
              mask_vector = mask_vector>mask_vector.mean()
              mask_ = mask_.view(-1,20,1,1)

              # apply the Soft Spatial Argmax to get the final prediction of the spatial coordinates of fixations
              result = self.soft_sam(out*mask_)/torch.tensor(out.shape[-2:]).float().cuda()
              
              # If the domain adaptation is enabled, use the domain classifier to predict the domain of the input
              if self.domain:   
                    reverse_features = RevGrad(Flatten()(adapt(bottel)), 1.0)
                    domain_out = self.domain_classifier(reverse_features)
                    return result ,  gaussian_maps, domain_out, mask_vector   
                
              return result, gaussian_maps, mask_vector     
            

# Test the model
if __name__ == '__main__' : 
    with torch.no_grad():
        model = ScanPathModel()
        model.cuda()

        ut1, _, _  = model(torch.rand([1,3,192, 256]).cuda())
        ut2, _, _  = model(torch.rand([1,3,192, 256]).cuda())

    print(ut1.shape)
    print(ut2.shape)

