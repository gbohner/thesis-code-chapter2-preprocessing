import torch
import math
import gpytorch
from gpytorch.random_variables import RandomVariable, MixtureRandomVariable


import math
from numbers import Number

import torch
from torch.distributions import constraints
# from torch.distributions.exp_family import ExponentialFamily
# from torch.distributions.utils import broadcast_all # Serious bug! - https://github.com/pytorch/pytorch/issues/11242

from quadrature_integrator_torch import QuadratureIntegratorTorch

from IPython.core.debugger import set_trace


import warnings

class MixtureRandomVariableWithSampler(gpytorch.random_variables.MixtureRandomVariable):
    def sample(self, n_samples=1):
        # Get representation
        rand_vars, weights = self.representation()

        # Sample from a categorical distribution
        sample_ids = torch.distributions.categorical.Categorical(probs=weights).sample((n_samples,))

        # Sample from the individual distributions
        samples = torch.tensor([rand_vars[i].sample(1) for i in sample_ids], device=weights.device)

        return samples
    
    

class BatchRandomVariable(gpytorch.random_variables.RandomVariable):
    def __init__(self, *rand_vars, **kwargs):
        """
        Batch of random variables
        Params:
        - rand_vars (iterable of RandomVariables with b elements)
        """
        
        super(BatchRandomVariable, self).__init__(*rand_vars, **kwargs)
        if not all(isinstance(rand_var, RandomVariable) for rand_var in rand_vars):
            raise RuntimeError("Everything needs to be an instance of a random variable")
            
        self.rand_vars = rand_vars
        
    def representation(self):
        return self.rand_vars
    
    def mean(self):
        means = [rand_var.mean() for rand_var in self.rand_vars]
        return torch.tensor(means, device=means[0].device)
    
    def var(self):
        variances = [rand_var.var() for rand_var in self.rand_vars]
        return torch.tensor(variances, device=variances[0].device)
    
    def sample(self, n_samples=1):
        '''
        Sample n_samples for each of the b rand_vars and return an 
        b x (d) x n_samples... object consistent with random variables for which batch mode is enabled
        '''
        
        # b x ... x n_samples  Implementation (copying GaussianRandomVariable)
        samples = torch.cat([rand_var.sample(n_samples).squeeze().unsqueeze(0) for rand_var in self.rand_vars])
        return samples

#         # n_samples x b x ... Implementation
#         samples = torch.cat([rand_var.sample(n_samples).unsqueeze(0) for rand_var in self.rand_vars])
#         return samples.permute(1,0, *range(2,samples.ndimension()))





def erfcx(x):
    #https://stackoverflow.com/questions/8962542/is-there-a-scaled-complementary-error-function-in-python-available
    ret1 = (1.-x[x<3.].erf()).mul(x[x<3.].pow(2.).exp())
    
    y = 1. / x[x>=3.]
    z = y * y
    s = y*(1.+z*(-0.5+z*(0.75+z*(-1.875+z*(6.5625-29.53125*z)))))
    ret2 = s * 0.564189583547756287
    
    ret_final = torch.zeros_like(x)
    ret_final[x<3.] += ret1
    ret_final[x>=3.] += ret2
    
    return ret_final


class ExponentiallyModifiedGaussian(torch.distributions.Distribution):
    """
    R. Dossi et al. / Nuclear Instruments and Methods in Physics Research A 451 (2000) 623}637
    Equation 9, with pE = 1.
    
    There's an error in the erf part which should read (x-xp) instead of just xp
    """
    def __init__(self, loc, scale, expamplitude, validate_args=None):
        #self.loc, self.scale, self.expamplitude = broadcast_all(loc, scale, expamplitude) # Serious bug - https://github.com/pytorch/pytorch/issues/11242
        self.loc = loc
        self.scale = scale
        self.expamplitude = expamplitude
        if isinstance(loc, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(ExponentiallyModifiedGaussian, self).__init__(batch_shape, validate_args=validate_args)
        
        self.integrator = QuadratureIntegratorTorch(40.)
        
    def log_prob(self, x):
        """
        For numerical stability we use multiple implementations as suggested on 
        https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution
        """
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        scale = self.scale
        var = scale.pow(2.)
        A = self.expamplitude
        xp = self.loc
        
        # For numerically stable wiki implementation
        sigma = scale
        mu = xp
        tau = A
        
        x_inp_dims = x.size()
        x = torch.tensor(x).to(scale.device).view([-1]+[1]*self.loc.ndimension())
        
        z = ((scale/A)-(x-xp).div(scale)).mul(torch.tensor(1./2., device=x.device).sqrt())

        #         print(z.max())
        #         print(z.min())


        # Compute all 3 wikipedia implementation, then fill in the result based on value of z

        # model 1
        #norm_part = sigma.div(tau).mul(torch.tensor(math.pi/2.).sqrt())
        #norm_part = torch.tensor(1.)
        norm_part = (2.*tau).reciprocal()

        exp_part = (var-2.*tau*(x[z<3.]-mu)).div(2.*tau.pow(2))
        erf_part = z[z<3.]

        ret1 = norm_part.log() + exp_part + (1.-erf_part.erf()).log()

        # model 2

        exp_part_approx = (x[(z>=3.)*(z<8191.)]-xp).pow(2).div(-2.*var)
        erfcx_part = z[(z>=3.)*(z<8191.)]

        ret2 = norm_part.log() + exp_part_approx + erfcx(erfcx_part).log()


        # model 3
        exp_part_approx_model3 = (x[z>=8191.]-xp).pow(2).div(-2.*var)
        div_part = 1.-(x[z>=8191.]-xp).mul(A).div(var)  

        ret3 = exp_part_approx_model3 - div_part.log()

        #set_trace()

        ret_final = torch.zeros_like(z)  
        ret_final[z<3.] += ret1
        ret_final[(z>=3.)*(z<8191.)] += ret2
        ret_final[z>=8191.] += ret3
        
        
        
        #ret_final = ret1*(z<3.).float() + ret2*(z<8191.).float()*(z>=3.).float() + ret3*(z>=8191.).float()  
        
        # This does not seem to work for backward pass:
#         ret_final = ret3   # Bad estimate for small-ish z, goes over 1 around 4
#         ret_final[z<8191.] = ret2[z<8191.] # based on single-precision float
#         ret_final[z<3.] = ret1[z<3.] # Experimentally, see below
        
        
        if self.loc.ndimension()>0:
            return ret_final.view(*(x_inp_dims + self.loc.size())).squeeze(-self.loc.ndimension()-1)
        else:
            return ret_final.view(*x_inp_dims)
        
        
        
    
        """
            # Check which part works best where - seems like the erfcx implementation is not very good
            return torch.stack([ret1[:,1,1], ret2[:,1,1], ret3[:,1,1]], dim=1), z[:,1,1]
            a = ExponentiallyModifiedGaussian(loc = 0.*torch.ones(3,4), 
                                      scale = 0.6*torch.ones(3,4), 
                                      expamplitude=.1*torch.ones(3,4))
            inp = torch.arange(-20., 30., 0.01)
            retval, z = a.log_prob(inp.view(-1,1,1))#[:,1,1]
            #print(retval)
            plot(retval.view(-1,3).exp().clamp(0.,1.), z.view(-1))
        """ 
            
    def cdf(self, x):
        if not isinstance(x, Number) or self.loc.ndimension()>0:
            raise NotImplementedError
        return self.integrator.to(self.loc.device).integrate(
            lambda tmp: self.log_prob(tmp).exp(), 
            a = (self.loc.min()-5.*self.scale.max()),
            b = x,
        )
#         return self.integrator.batch_integrate(
#             lambda tmp: self.log_prob(tmp).exp(), 
#             a = -10.,
#             b = 10.,
# #             a = (self.loc.min()-2*self.scale.max()),
# #             b = x,
#             viewAs = [-1]+[1]*self.loc.ndimension())


class SingleElectronResponse(torch.distributions.Distribution):
    def __init__(self, 
                 loc, scale, 
                 pedestal_loc, pedestal_scale, 
                 underamplified_amplitude, underamplified_probability):
        self.ExpModGauss = ExponentiallyModifiedGaussian(loc=pedestal_loc, 
                                                         scale=pedestal_scale, 
                                                         expamplitude=underamplified_amplitude)
        self.Normal = torch.distributions.Normal(loc = loc, 
                                                 scale=scale)
        self.underamplified_probability = underamplified_probability
        
    def log_prob(self, x):
        return (
            self.underamplified_probability * self.ExpModGauss.log_prob(x).exp() +
            (1.-self.underamplified_probability) * self.Normal.log_prob(x).exp()
        ).log()
    
    def cdf(self, x):
        return (
            self.underamplified_probability * self.ExpModGauss.cdf(x) +
            (1.-self.underamplified_probability) * self.Normal.cdf(x)
        )
    
    
class SingleElectronResponseRandomVariable(gpytorch.random_variables.RandomVariable):
    def __init__(self, 
                 loc, scale, 
                 pedestal_loc, pedestal_scale, 
                 underamplified_amplitude, underamplified_probability):
        self.distribution = SingleElectronResponse(loc, scale, 
                 pedestal_loc, pedestal_scale, 
                 underamplified_amplitude, underamplified_probability)                   
        
        # For higher photon counts we ignore convolution with the noise and compute moments of idealised SER0
     
    def representation(self):
        return self.distribution
        
    def mean(self):
        pE = self.distribution.underamplified_probability.squeeze()
        A = self.distribution.ExpModGauss.expamplitude.squeeze()
        gain = self.distribution.Normal.loc.squeeze()
        s0 = self.distribution.Normal.scale.squeeze()
        
        return (pE*A + (1.-pE)*gain)                   
        
    
    def var(self):
        pE = self.distribution.underamplified_probability.squeeze()
        A = self.distribution.ExpModGauss.expamplitude.squeeze()
        gain = self.distribution.Normal.loc.squeeze()
        s0 = self.distribution.Normal.scale.squeeze()
        
        return (pE*2.*A.pow(2.) + (1.-pE)*(gain.pow(2.)+s0.pow(2.)) - self.mean().pow(2.))
        
    def sample(self, n_samples=1, n_categories=int(100), oversample_extremes = False):
        """approximates the distribution as a categorical binned distribution"""
        
        bin_mids = torch.linspace(
            float(min((self.distribution.ExpModGauss.loc.min()-4.*self.distribution.ExpModGauss.scale.max()),
                    self.distribution.Normal.loc.min()-2.*self.distribution.Normal.scale.max())),
            float(self.distribution.Normal.loc.max()+2.*self.distribution.Normal.scale.max()),
            n_categories
        ).to(self.distribution.Normal.loc.device)
               
        bin_size = bin_mids[1] - bin_mids[0]
               
        weights = torch.zeros_like(bin_mids)
        weights = self.distribution.log_prob(bin_mids).exp()
        if oversample_extremes:
            weights[0] = self.distribution.cdf(float(bin_mids[0]))
            weights[-1] = 1.-self.distribution.cdf(float(bin_mids[-1]))
            
        weights = weights.div(weights.sum())
            
               
        # Sample from a categorical distribution
        sample_ids = torch.distributions.categorical.Categorical(probs=weights).sample((n_samples,)
                                                                                      ).to(self.distribution.Normal.loc.device)
               
        # Sample from the individual distributions
        samples = torch.tensor([bin_mids[i]+torch.rand(1, device=weights.device)*bin_size - bin_size/2. for i in sample_ids], device=weights.device)

        return samples
    
    
    
class Normal(torch.distributions.exp_family.ExponentialFamily):
    """
    broadcast_all is bugged, so reimplementing distributions I use without it
    
    Creates a normal (also called Gaussian) distribution parameterized by
    `loc` and `scale`.

    Example::

        >>> m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # normally distributed with loc=0 and scale=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the distribution
            (often referred to as sigma)
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)

    def __init__(self, loc, scale, validate_args=None):
        
        self.loc = loc
        self.scale = scale
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(Normal, self).__init__(batch_shape, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.normal(self.loc.expand(shape), self.scale.expand(shape))


    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = self.loc.new(shape).normal_()
        return self.loc + eps * self.scale


    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = (self.scale ** 2)
        log_scale = math.log(self.scale) if isinstance(self.scale, Number) else self.scale.log()
        return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))


    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 0.5 * (1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2)))


    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.loc + self.scale * torch.erfinv(2 * value - 1) * math.sqrt(2)


    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)


    @property
    def _natural_params(self):
        return (self.loc / self.scale.pow(2), -0.5 * self.scale.pow(2).reciprocal())

    def _log_normalizer(self, x, y):
        return -0.25 * x.pow(2) / y + 0.5 * torch.log(-math.pi / y)
        
    
    
        