from typing import Union
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.common_types import _size_2_t
import torch.nn.functional as F

class P_Conv2d(nn.Conv2d):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: _size_2_t | str = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)

        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        if self.bias is not None:
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))
        
    def forward(self, input: Tensor) -> Tensor:
        bias = None
        if self.bias is not None:
            bias = self.bias * self.bias_mask
        return self._conv_forward(input, self.weight * self.weight_mask, bias)
    
def rm_modules(model):
    rm_blocks = []

    #Backbone modules
    rm_blocks += [(model.model[0].conv,  'weight')]

    rm_blocks += [(model.model[1].conv,  'weight')]

    rm_blocks += [(model.model[2].cv1.conv,  'weight')]
    rm_blocks += [(model.model[2].cv2.conv,  'weight')]
    rm_blocks += [(model.model[2].m[i].cv1.conv,  'weight') for i in range(len(model.model[2].m))]
    rm_blocks += [(model.model[2].m[i].cv2.conv,  'weight') for i in range(len(model.model[2].m))]
    
    rm_blocks += [(model.model[3].conv,  'weight')]

    rm_blocks += [(model.model[4].cv1.conv,  'weight')]
    rm_blocks += [(model.model[4].cv2.conv,  'weight')]
    rm_blocks += [(model.model[4].m[i].cv1.conv,  'weight') for i in range(len(model.model[4].m))]
    rm_blocks += [(model.model[4].m[i].cv2.conv,  'weight') for i in range(len(model.model[4].m))]
    
    rm_blocks += [(model.model[5].conv,  'weight')]

    rm_blocks += [(model.model[6].cv1.conv,  'weight')]
    rm_blocks += [(model.model[6].cv2.conv,  'weight')]    
    rm_blocks += [(model.model[6].m[i].cv1.conv,  'weight') for i in range(len(model.model[6].m))]
    rm_blocks += [(model.model[6].m[i].cv2.conv,  'weight') for i in range(len(model.model[6].m))]
    
    rm_blocks += [(model.model[7].conv,  'weight')]

    rm_blocks += [(model.model[8].cv1.conv,  'weight')]
    rm_blocks += [(model.model[8].cv2.conv,  'weight')]    
    rm_blocks += [(model.model[8].m[i].cv1.conv,  'weight') for i in range(len(model.model[8].m))]
    rm_blocks += [(model.model[8].m[i].cv2.conv,  'weight') for i in range(len(model.model[8].m))]

    rm_blocks += [(model.model[9].cv1.conv,  'weight')]
    rm_blocks += [(model.model[9].cv2.conv,  'weight')]  

    # Head modules
    rm_blocks += [(model.model[12].cv1.conv,  'weight')]
    rm_blocks += [(model.model[12].cv2.conv,  'weight')]  
    rm_blocks += [(model.model[12].m[i].cv1.conv,  'weight') for i in range(len(model.model[12].m))]
    rm_blocks += [(model.model[12].m[i].cv2.conv,  'weight') for i in range(len(model.model[12].m))]

    rm_blocks += [(model.model[15].cv1.conv,  'weight')]
    rm_blocks += [(model.model[15].cv2.conv,  'weight')]  
    rm_blocks += [(model.model[15].m[i].cv1.conv,  'weight') for i in range(len(model.model[15].m))]
    rm_blocks += [(model.model[15].m[i].cv2.conv,  'weight') for i in range(len(model.model[15].m))]

    rm_blocks += [(model.model[16].conv,  'weight')]

    rm_blocks += [(model.model[18].cv1.conv,  'weight')]
    rm_blocks += [(model.model[18].cv2.conv,  'weight')]  
    rm_blocks += [(model.model[18].m[i].cv1.conv,  'weight') for i in range(len(model.model[18].m))]
    rm_blocks += [(model.model[18].m[i].cv2.conv,  'weight') for i in range(len(model.model[18].m))]

    rm_blocks += [(model.model[19].conv,  'weight')]

    rm_blocks += [(model.model[21].cv1.conv,  'weight')]
    rm_blocks += [(model.model[21].cv2.conv,  'weight')]  
    rm_blocks += [(model.model[21].m[i].cv1.conv,  'weight') for i in range(len(model.model[21].m))]
    rm_blocks += [(model.model[21].m[i].cv2.conv,  'weight') for i in range(len(model.model[21].m))]
    
    '''
    rm_blocks += [(model.model[22].cv2[0][0].conv, 'weight')]
    rm_blocks += [(model.model[22].cv2[0][1].conv, 'weight')]
    rm_blocks += [(model.model[22].cv2[0][2],  'weight')]
    rm_blocks += [(model.model[22].cv2[1][0].conv, 'weight')]
    rm_blocks += [(model.model[22].cv2[1][1].conv, 'weight')]
    rm_blocks += [(model.model[22].cv2[1][2],  'weight')]
    rm_blocks += [(model.model[22].cv2[2][0].conv, 'weight')]
    rm_blocks += [(model.model[22].cv2[2][1].conv, 'weight')]
    rm_blocks += [(model.model[22].cv2[2][2],  'weight')]
    rm_blocks += [(model.model[22].cv3[0][0].conv, 'weight')]
    rm_blocks += [(model.model[22].cv3[0][1].conv, 'weight')]
    rm_blocks += [(model.model[22].cv3[0][2],  'weight')]
    rm_blocks += [(model.model[22].cv3[1][0].conv, 'weight')]
    rm_blocks += [(model.model[22].cv3[1][1].conv, 'weight')]
    rm_blocks += [(model.model[22].cv3[1][2],  'weight')]
    rm_blocks += [(model.model[22].cv3[2][0].conv, 'weight')]
    rm_blocks += [(model.model[22].cv3[2][1].conv, 'weight')]
    rm_blocks += [(model.model[22].cv3[2][2],  'weight')]
    '''
    #rm_blocks += [(model.model[22].dfl.conv, 'weight')]

    return rm_blocks

#def initialize()
def magnitude(rm_modules, sparsity):

    scores = [rm_modules[i][0].weight.detach().cpu().clone().abs().flatten() for i in range(len(rm_modules))]    
    scores = torch.cat(scores)
    masks = [rm_modules[i][0].weight_mask.detach().cpu().clone().abs().flatten() for i in range(len(rm_modules))]
    masks = torch.cat(masks)
    scores = scores * masks
    pre_sparsity = 1 - masks.count_nonzero() / masks.numel()
    print(pre_sparsity)
    n_params_to_prune = int(sparsity * masks.count_nonzero() + masks.numel() - masks.count_nonzero()) if pre_sparsity != 0 else int(sparsity * scores.__len__())
    print(n_params_to_prune)
    k = torch.kthvalue(scores, n_params_to_prune)[0]
    #print(k)

    for i in range(len(rm_modules)):
        rm_modules[i][0].weight_mask = torch.where(((rm_modules[i][0].weight_mask * rm_modules[i][0].weight.abs().detach()) <= k).to(dtype=torch.bool), 0, 1)

    p = 0
    for i in range(len(rm_modules)):
        p += rm_modules[i][0].weight_mask.count_nonzero()
    print(p / scores.numel())

def snip(rm_modules, sparsity, scores):

    scores = torch.cat(scores)
    print(scores)
    masks = [rm_modules[i][0].weight_mask.detach().cpu().clone().abs().flatten() for i in range(len(rm_modules))]
    masks = torch.cat(masks)
    scores = scores * masks
    pre_sparsity = 1 - masks.count_nonzero() / masks.numel()
    print(pre_sparsity)
    n_params_to_prune = int(sparsity * masks.count_nonzero() + masks.numel() - masks.count_nonzero()) if pre_sparsity != 0 else int(sparsity * scores.__len__())
    print(n_params_to_prune)
    k = torch.kthvalue(scores, n_params_to_prune)[0]
    print(k)
    n = 0
    for i in range(len(rm_modules)):
        m = rm_modules[i][0].weight_mask.numel()
        s = scores[n:n+m]
        rm_modules[i][0].weight_mask = torch.where(((rm_modules[i][0].weight_mask * s.reshape(rm_modules[i][0].weight_mask.shape).to(rm_modules[i][0].weight_mask.device)) <= k).to(dtype=torch.bool), 0, 1)
        n += m

    p = 0
    for i in range(len(rm_modules)):
        p += rm_modules[i][0].weight_mask.count_nonzero()
    print(p / scores.numel())