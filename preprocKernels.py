import torch
import math
import gpytorch

from torch import nn, optim
from gpytorch.kernels import Kernel, WhiteNoiseKernel

import warnings
from IPython.core.debugger import set_trace



import preprocUtils


class MexicanHatKernel(Kernel):
    def __init__(self, lengthscale=1.):
        super(MexicanHatKernel, self).__init__()
        self.register_parameter(
            'log_lengthscale',
            preprocUtils.toTorchParam(lengthscale, ndims=1, to_log=True),
            prior = gpytorch.priors.SmoothedBoxPrior(-2.,1.,sigma=0.1)
        )
    def forward(self, x1, x2):
        distance = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).abs()  # distance = x^(i) - z^(i)

        exp_term = distance.div(self.log_lengthscale.exp()).pow_(2).mul(-1.)
        cos_term = distance.div(self.log_lengthscale.exp()).mul_(math.pi)
        res = exp_term.exp_() * cos_term.cos_()

        return res.view(x1.size(0), x1.size(1), x2.size(1))


from gpytorch.lazy import DiagLazyVariable, ZeroLazyVariable
class WhiteNoiseKernelBugfix(WhiteNoiseKernel):
    """
    Fixes the issue where WhiteNoiseKernel would return zeros instead of the var * I out of training
    """       
    def forward(self, x1, x2):
        if self.training and torch.equal(x1, x2):
                # Reshape into a batch of batch_size diagonal matrices, each of which is
                # (data_size * task_size) x (data_size * task_size)
                return DiagLazyVariable(self.variances.squeeze().expand(x1.size(-2)).view(1,-1))
        elif x1.size(-2) == x2.size(-2) and (x1.size(-2) == self.variances.numel() or self.variances.numel()==1) and torch.equal(x1, x2):
                return DiagLazyVariable(self.variances.squeeze().expand(x1.size(-2)).view(1,-1))
        else:
            set_trace()
            return ZeroLazyVariable(x1.size(-3), x1.size(-2), x2.size(-2))

class ScaleKernel(Kernel):
    """
    Adds an extra tunable output-scale parameter to the kernel
    """
    def __init__(self, base_kernel, weight=1.0, weight_prior=None, fix_scale = False):
        super(ScaleKernel, self).__init__()
        self.base_kernel = base_kernel
        self.register_parameter(
            'log_outputscale',
            preprocUtils.toTorchParam(weight, ndims=1, to_log=True),
            prior = weight_prior if weight_prior is not None else gpytorch.priors.SmoothedBoxPrior(math.exp(-3),math.exp(1),log_transform=True)
        )
        
        if fix_scale:
            self.log_outputscale.requires_grad = False
            
    def forward(self, *inputs, **kwargs):
        # Right-multiple (due to lazy-variables) with output scale
        return self.base_kernel(*inputs, **kwargs).mul(self.log_outputscale.exp())


class SymmetriseKernelRadially(Kernel):
    """
    This class may be applied to any other Kernel, and it will modify it's input such that 
    the base kernel only receives the euclidean distance from the center for each input point
    
    __init__(self, base_kernel_module, center, active_dims=None)
    """
    def __init__(self, base_kernel_module, center, active_dims=None):
        super(SymmetriseKernelRadially, self).__init__(active_dims=active_dims)
        self.base_kernel_module = base_kernel_module
        
        self.register_parameter(name="center", 
                                parameter=torch.nn.Parameter(center.squeeze()),
                                # prior is assuming initial center is around [0.5]*ndims, if center is at [0,0] this is bad
                                prior=gpytorch.priors.SmoothedBoxPrior(
                                        0.25*center*2,  
                                        0.75*center*2, 
                                        sigma=0.05**(center.numel()), log_transform=False)
                               )
        
        # Store the minimum and the width to scale kernel to data
        self.register_buffer("center_x_mins", torch.zeros_like(self.center))
        self.register_buffer("center_x_widths", torch.ones_like(self.center))
        
        
        # TODO: Check if base_kernel_module is defined for 1-dimensional inputs (that our radial transform creates)
        
    def forward(self, x1, x2, **kwargs):
        """
        Seemingly kernel inputs need to be 3 dimensional ( ? x num_points x num_dimensions),
        if any of these dimensions are dropped it doesn't work. ? is usually 1, I imagine it is num_batch
        """
        if not self.center.shape[0] == x1.shape[-1]:
            #print(x1.shape)
            #print(self.center.shape)
            warnings.warn("The input dimension should be the same as the center point dimension")
            #raise RuntimeError("The input dimension should be the same as the center point dimension")
            
        #set_trace()
        
        x_center = self.center*self.center_x_widths + self.center_x_mins
        
        return self.base_kernel_module(
            (x1-x_center.view(1,1,-1)).pow(2).sum(-1).sqrt().unsqueeze(-1), 
            (x2-x_center.view(1,1,-1)).pow(2).sum(-1).sqrt().unsqueeze(-1), 
            **kwargs
        ).evaluate()

class SymmetriseKernelLinearly(Kernel):
    """
    This class may be applied to any other Kernel, and it will modify it's input such that 
    the base kernel only receives the absolute vector distance from the center for each input point
    
    __init__(self, base_kernel_module, center, active_dims=None)
    """
    def __init__(self, base_kernel_module, center, active_dims=None):
        super(SymmetriseKernelLinearly, self).__init__(active_dims=active_dims)
        self.base_kernel_module = base_kernel_module
        
        self.register_parameter(name="center", 
                                parameter=torch.nn.Parameter(center.squeeze()),
                                # prior is assuming initial center is around [0.5]*ndims, if center is at [0,0] this is bad
                                prior=gpytorch.priors.SmoothedBoxPrior(
                                        0.25*center*2,  
                                        0.75*center*2, 
                                        sigma=0.05**(center.numel()), log_transform=False)
                               )
        
        # Store the minimum and the width to scale kernel to data
        self.register_buffer("center_x_mins", torch.zeros_like(self.center))
        self.register_buffer("center_x_widths", torch.ones_like(self.center))
        
        # TODO: Check if base_kernel_module is defined on the same dimensions as center
        
    def forward(self, x1, x2, **kwargs):
        if not self.center.shape[0] == x1.shape[-1]:
            raise RuntimeError("The input dimension should be the same as the center point dimension")
            
        x_center = self.center*self.center_x_widths + self.center_x_mins
            
        return self.base_kernel_module(
            (x1 - x_center.view(1,1,-1)).abs(), 
            (x2 - x_center.view(1,1,-1)).abs(), 
            **kwargs
        )