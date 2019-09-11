import torch
import math
import gpytorch

from gpytorch.functions import add_diag
from gpytorch.likelihoods import Likelihood
from gpytorch.priors._compatibility import _bounds_to_prior
from gpytorch.random_variables import GaussianRandomVariable, MixtureRandomVariable, RandomVariable  
from gpytorch.priors import SmoothedBoxPrior

import preprocUtils
import preprocRandomVariables

from preprocUtils import toTorchParam
from preprocRandomVariables import BatchRandomVariable, MixtureRandomVariableWithSampler
from preprocRandomVariables import SingleElectronResponse, SingleElectronResponseRandomVariable
from preprocRandomVariables import Normal # Use my custom Normal that doesn't use the buggy torch.distributions.utils.broadcast_all



from quadrature_integrator_torch import QuadratureIntegratorTorch

from IPython.core.debugger import set_trace



# def probGaussTorch(y, mu, sig2):
#     return torch.exp(-0.5*(y-mu)**2/sig2 - 0.5*torch.log(2*np.pi*sig2))

# def probPoissonTorch(y, rate):
#     return torch.exp(torch.log(rate)*y - rate - torch.lgamma(y+1))






def getVar(latent_func):
    """
    Use this method to get variance approximations of interpolated lazy variables
    """
    if latent_func.covar().size(0) > 2000 and isinstance(latent_func.covar(), gpytorch.lazy.InterpolatedLazyVariable):
        return latent_func.covar()._approx_diag()
    else:
        return latent_func.var()


class BasePhotomultiplierLikelihood(Likelihood):
    """
    Defines useful functions for various photomultiplier models
    """
    
    def __init__(self, 
                 gain=None, offset=None, noise=None,
                 gaussQuadratureDegree = int(10)):
        super(BasePhotomultiplierLikelihood, self).__init__()

        
        #### Add a Gauss-Hermite integrator, we're going to need it all the time
        self.integrator = QuadratureIntegratorTorch(gaussQuadratureDegree)
        

        #### -----------------------------------
        #### Register photomultiplier parameters
        #### -----------------------------------

        self.register_parameter(name="log_gain", 
                                parameter=toTorchParam(gain if gain is not None else 1., ndims=1, to_log=True), 
                                prior=SmoothedBoxPrior(-3, 8, sigma = 0.1))
        self.register_parameter(name="offset", 
                                parameter=toTorchParam(offset if offset is not None else 0., ndims=1, to_log=False), 
                                prior=SmoothedBoxPrior(-200, 200, sigma = 5.0))        
        self.register_parameter(name="log_noise", 
                                parameter=toTorchParam(noise if noise is not None else 1., ndims=1, to_log=True), 
                                prior=SmoothedBoxPrior(-8, 20, sigma = 0.1))


    def single_log_prob(self, inp, cur_target):
        """ Required for self.log_probability(), 
            takes a (N, ) vectors inp [concrete realisation of input] and cur_target and 
            return log p(cur_target | inp)
        """
        raise NotImplementedError

    
    def log_probability_each(self, latent_func, target):
        """
        Compute the expectation 
                E_f [log p(y|f) ] = \integral (log p(p|f)) * p(f | mean, var) df
                
        One might average over a set of latent function samples
        For the purposes of our variational inference implementation, y is an
        n-by-1 label vector, and f is an n-by-s matrix of s samples from the
        variational posterior, q(f|D).
        """
        input_device = latent_func.mean().device
        assert(input_device == target.device)
        
        if target.dim()==1:
            target = target.unsqueeze(1)
        
        # Define the log probability function, 
        # then Gauss-Hermite integrate it using the inherited self.integrator
        latStd = getVar(latent_func).sqrt()

        res = torch.zeros(target.size(0), device=input_device)
        
        for i in range(target.size(1)):
            target_nonans = target[:,i]
            pred_nonans_mean = latent_func.mean()[(torch.isnan(target_nonans)==False)]
            pred_nonans_std = latStd[(torch.isnan(target_nonans)==False)]
            target_nonans = target_nonans[(torch.isnan(target_nonans)==False)]
            
            tmp=self.integrator.batch_integrate_gauss(
                lambda x: self.single_log_prob(x, target_nonans),
                mu = pred_nonans_mean,
                sig = pred_nonans_std
           )
            
#             tmp=self.integrator.batch_integrate_gauss(
#                 lambda x: self.single_log_prob(x, target[:,i]),
#                 mu = latent_func.mean(),
#                 sig = latStd
#            )
            
            res+= tmp
            del tmp
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return res.div(target.size(1))
    
    
    
    def log_probability(self, latent_func, target):
        """
        Compute the expectation 
                E_f [log p(y|f) ] = \integral (log p(y|f)) * p(f | mean, var) df
                
        One might average over a set of latent function samples
        For the purposes of our variational inference implementation, y is an
        n-by-1 label vector, and f is an n-by-s matrix of s samples from the
        variational posterior, q(f|D).
        """
        input_device = latent_func.mean().device
        assert(input_device == target.device)
        
        if target.dim()==1:
            target = target.unsqueeze(1)
        
        # Define the log probability function, 
        # then Gauss-Hermite integrate it using the inherited self.integrator
        latStd = getVar(latent_func).sqrt()

        res = 0
        for i in range(target.size(1)):
            target_nonans = target[:,i]
            pred_nonans_mean = latent_func.mean()[(torch.isnan(target_nonans)==False)]
            pred_nonans_std = latStd[(torch.isnan(target_nonans)==False)]
            target_nonans = target_nonans[(torch.isnan(target_nonans)==False)]
            
            tmp=self.integrator.batch_integrate_gauss(
                lambda x: self.single_log_prob(x, target_nonans),
                mu = pred_nonans_mean,
                sig = pred_nonans_std
           ).sum() # This sum is over different x locations
            
#             tmp=self.integrator.batch_integrate_gauss(
#                 lambda x: self.single_log_prob(x, target[:,i]),
#                 mu = latent_func.mean(),
#                 sig = latStd
#            ).sum() # This sum is over different x locations
            
            res+= tmp # This sum is essentially over different time points (i in target.size(1))
            del tmp
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return res.div(target.size(1))
        
    def forward(self, *inputs, **kwargs):
        """
        Compute the expectation
            p(y|x) = E_f [ p(y | f ) * p (f | x)]
        
        Computes a predictive distribution p(y*|x*) given either a posterior
        distribution p(f|D,x) or a prior distribution p(f|x) as input.
        With both exact inference and variational inference, the form of
        p(f|D,x) or p(f|x) should usually be Gaussian. As a result, input
        should usually be a GaussianRandomVariable specified by the mean and
        (co)variance of p(f|...).
        """
        raise NotImplementedError
        
    def Poisson_log_prob(self, rate, value):
        return (rate.log() * value) - rate - (value + 1).lgamma()

    def getPhotonLogProbs(self, poissMeans, max_photon=30., reNormalise = False):
        """
        Given an array of poisson means (Pdims) and a maximum photon count (D), 
        return 
         - the counts vector (D x [1]*len(Pdims)-2) and the 
         - the log probabilities matrix (D x Pdims)
        """

        # Check if we do not need to represent counts up to max_photon, or we'd need more:
        # TODO - check poissMeans.max() and it's CDF value at max_photon, 
            # if too low - warn for more, it too high, just cut from max_photon
            
        # Average over log probability of any given draw from the poisson (max max_photons)
        photon_counts = torch.arange(max_photon, device=poissMeans.device).view(*([-1]+[1]*poissMeans.ndimension()))
        photon_log_probs = self.Poisson_log_prob(poissMeans.unsqueeze(0), photon_counts)

        if reNormalise:
            photon_log_probs -= photon_log_probs.logsumexp(0).unsqueeze(0)

        return photon_counts, photon_log_probs
    
    
    
        
    
    
class LinearGainLikelihood(BasePhotomultiplierLikelihood):
    """
    Assumes that the incoming continuous (!) log number gets linearly multiplied by log_gain.exp(),
    and the observed value is a scaled Poisson distribution with a given output-offset (instead of the usual input-offset)
    approximated by an appropriate Gaussian
    
    
    p(y | f) = N_y ( g*f + offset,  g^2 * f^2 + sigma_y^2 )
    
        # noise = sigma_y^2
    """
    def __init__(self, gain=None, offset=None, noise=None): 
        super(LinearGainLikelihood, self).__init__(gain = gain, offset = offset, noise=noise)
        
    def forward(self, latent_func):
        
        pred_mean = (latent_func.mean() + self.log_gain).exp() + self.offset
        pred_var = (self.log_gain.exp().pow(2) * 
                        (latent_func.mean().exp().pow(2) + 2*(getVar(latent_func).exp().pow(2))))
             # g^2 * (mean^2 + 2 * var^2) - expected log prob
            
        return GaussianRandomVariable(pred_mean, gpytorch.lazy.DiagLazyVariable(pred_var))
    
    def single_log_prob(self, inp, cur_target):
        """
        Input is directly from latent_func, means it still needs to be exponentiated before multiplying
        """
        #set_trace()
        prob_var = self.log_noise.exp() + (inp + self.log_gain).exp()**2
        res = -0.5 * ((cur_target - ((inp + self.log_gain).exp() + self.offset) )  ** 2) / (
            prob_var
        )
        res += -0.5 * prob_var.log() - 0.5 * math.log(2 * math.pi)
        return res
    
    
#     def log_probability(self, latent_func, target):
#         """
#         Note that latent_func represent the logarithm of the incoming signal!
        
        
#         """
#         input_device = latent_func.mean().device
#         assert(input_device == target.device)
        
#         if target.dim()==1:
#             target = target.unsqueeze(1)
        
#         # Define the log probability function, 
#         # then Gauss-Hermite integrate it using the inherited self.integrator
        
        
#         latStd = getVar(latent_func).sqrt()

#         res = 0
#         for i in range(target.size(1)):
#             res+=self.integrator.batch_integrate_gauss(
#                 lambda x: self.single_log_prob(x, target[:,i]),
#                 mu = latent_func.mean(),
#                 sig = latStd
#            ).sum()
        
#         return res.div(target.size(1))
        
        """
        res = 0.
        for i in range(target.size(1)):
            res += self.single_log_prob(latent_func.mean(), target[:,i]).sum()
        
        
        return res

        """
        
        
        """
        def toApply(pred_row):
            #pred_row is [latent mean, latent std, target]
            #set_trace()
            return self.integrator.integrate_gauss(
                lambda x: single_log_prob(x, pred_row[2]),
                mu = pred_row[0],
                sig = pred_row[1]
            )
        
        eachIntegral = [
            preprocUtils.apply(
                lambda x : toApply(x), 
                torch.stack([latent_func.mean(), 
                          getVar(latent_func).sqrt(), 
                          target[:,i]],
                          dim=1),
                dim = 0
            ).sum()
            
            for i in range(target.size(1))]

        return torch.tensor(eachIntegral, device=input_device).sum()
        """        



class PoissonInputPhotomultiplierLikelihood(BasePhotomultiplierLikelihood):
    """
    Assumes that the incoming discrete number gets linearly multiplied by log_gain.exp(),
    and the observed value is a Normal distribution with a given input-offset, 
    whose mean and variance comes from the observed count.
    
    There is also an explicit "pedestal", which models the distribution around 0 counts
    
    p(f1 | f) = Poisson(f)
    p(y | f1) = {
        N_y ( g*f1 + offset,  g^2 * sigma_y^2 ) if f >0
        N_y (  offset , sigma_y_0^2)            if f == 0
    
        # noise = sigma_y^2
        # noise_pedestal = sigma_y_0^2
    """
    def __init__(self, gain=None, offset=None, noise=None, noise_pedestal=None): 
        super(PoissonInputPhotomultiplierLikelihood, self).__init__(gain = gain, offset = offset, noise=noise)
        
        self.register_parameter(name="log_noise_pedestal", 
                                parameter=toTorchParam(noise_pedestal if noise_pedestal is not None else 1e-2, ndims=1, to_log=True), 
                                prior=SmoothedBoxPrior(-10, 15, sigma = 0.1))
        
#     def forward(self, latent_func):
        
#         pred_mean = (latent_func.mean() + self.log_gain).exp() + self.offset
#         pred_var = (self.log_gain.exp().pow(2) * 
#                         (latent_func.mean().exp().pow(2) + 2*(getVar(latent_func).exp().pow(2))))
#              # g^2 * (mean^2 + 2 * var^2) - expected log prob
            
#         return GaussianRandomVariable(pred_mean, gpytorch.lazy.DiagLazyVariable(pred_var))
    
    
    def getLogProbSumOverTargetSamples(self, p_PM, cur_target_slice):
            """ 
            As we are supposedly dealing with truncated normals, we need to replace
            log probabilities of below pedestal values in the cur_target with the (log) CDF at the pedestal instead of the log_prob
            
            allLogProbs is N locations x max_photon photon counts array 
            """
            allLogProbs = p_PM.log_prob(cur_target_slice.view(-1,1))
            
            # Correct for observations less than the pedestal with log CDF instead of log_prob
            cutoff_point = torch.max(self.offset - self.log_noise_pedestal.exp().div(2.), self.offset.data.new([0.])).data
            
            ind_array = (cur_target_slice <= cutoff_point)
            if (ind_array).sum()>0:
                # Treat these as having near 0 probability of being more than a 0 photon count (not exactly 0 due to numerical considerations)
                allLogProbs[ind_array, 0] = self.offset.data.new([1.-1e-6]).log()
                allLogProbs[ind_array, 1:] = self.offset.data.new([1e-6]).div(allLogProbs.size(-1)).log()
                
                #allLogProbs[ind_array, :] = p_PM.cdf(torch.max(self.offset, torch.tensor(0.)).data).log().squeeze()
                #allLogProbs[cur_target_slice<1., :] = p_PM.cdf(cur_target_slice[cur_target_slice<1.].view(-1,1)).log()
            
            #set_trace()
            
            # Sum over target samples
            return allLogProbs.unsqueeze(1).sum(-2).squeeze() # This should work for 1d or 2d targets
    
    
    def createResponseDistributions(self, photon_counts):
        # p_PM is going to be a list here, representing a Gaussian mixture:
        #   p_PM[0] = noise distribution
        #   p_PM[1] = scaled gaussians
        
        p_PM = Normal(loc=(photon_counts*self.log_gain.exp()).squeeze()+self.offset, 
                                          scale=((photon_counts*(self.log_noise.mul(0.5).exp())).squeeze())
                                         )

        # Add the pedestal noise
        p_PM.scale[0] += self.log_noise_pedestal.mul(0.5).exp().squeeze()
        
        return p_PM
    
    
    def single_log_prob(self, inp, cur_target, batchsize = int(500), max_photon=25.):
        """
        Input is directly from latent_func, means it still needs to be exponentiated before multiplying.
        Input is the log mean of a Poisson distribution
        
        Input shape is [QuadratureWeights, N]
        cur_target is [N,]
        """
        
        # Have to do this in mini-batches as the resulting QuadWeights x MaxPhotons x N array is too big
        all_res = torch.zeros_like(inp)
        
        photon_counts, photon_log_probs = (
                self.getPhotonLogProbs(inp[:,:2].exp(), max_photon=float(max_photon), reNormalise = False))

        p_PM = self.createResponseDistributions(photon_counts)
        
        #set_trace()
        
        
        for i in range(int(int(inp.size(1))/batchsize)+1):
            cur_slice = slice(i*batchsize,min((i+1)*batchsize, inp.size(1)))
            if cur_slice.start == cur_slice.stop:
                break
            inp_cur = inp[:,cur_slice]
            if inp_cur.ndimension()==1:
                inp_cur = inp_cur.unsqueeze(1)
            
            photon_counts, photon_log_probs = (
                self.getPhotonLogProbs(inp_cur.exp(), max_photon=float(max_photon), reNormalise = False))

            ### return photon_counts, photon_log_probs, p_PM

            log_prob_sum_over_target_samples = self.getLogProbSumOverTargetSamples(p_PM, cur_target[cur_slice])
            if log_prob_sum_over_target_samples.ndimension()==1: # Correct for if we only get a single input value
                log_prob_sum_over_target_samples = log_prob_sum_over_target_samples.unsqueeze(0)



            all_res[:,cur_slice] += (log_prob_sum_over_target_samples.permute(1,0).unsqueeze(1) + photon_log_probs).logsumexp(dim=0)
            
            #del log_prob_sum_over_target_samples, photon_counts, photon_log_probs, p_PM
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return all_res # Same dimensionality as inp [QuadratureWeights, N]
    
      
    def forward(self, latent_func, approx=True, max_photon=int(20), batchsize = int(1000)):
        """
        Computes predictive distributions p(y|x) given a latent distribution
        p(f|x). To do this, we solve the integral:
            p(y|x) = \int p(y|f)p(f|x) df
            
        As the true representation gets storage expensive for large latent_func dimensionalities, 
        we provide an approximate method by default in which we only store the true mean and variance for each output
        """
        input_device = latent_func.mean().device
        
        if approx is False: ## Return the full Batch of MixtureRandomVariables (huge object, use only for few predictions)
            
            # Get the input standard deviation
            latStd = getVar(latent_func).sqrt()
            
            # Get the photon log probabilities
            photon_log_probs = self.integrator.batch_integrate_gauss(
                    lambda x: self.getPhotonLogProbs(x.exp(), max_photon=float(max_photon), reNormalise = False)[1].permute(1,0,2),
                    mu = latent_func.mean(),
                    sig = latStd,
                    viewAs = [-1, 1, 1]
               )
            
            # As these probabilities are a result of non-perfect integration, we need to renormalise them outside of the integration
            photon_log_probs -= photon_log_probs.logsumexp(0).view(1,-1)
            
            # For very low probabilities, the renomalisation may not have been perfect, renormalise once more in probability space
            photon_probs = photon_log_probs.exp()
            photon_probs = photon_probs.div(photon_probs.sum(0).unsqueeze(0))
            
            
            
            # Underlying mixture elements
            p_PM_each = [GaussianRandomVariable(self.offset, self.log_noise_pedestal.exp().diag())]
            p_PM_each.extend([
                GaussianRandomVariable(i*self.log_gain.exp()+self.offset, 
                                   i*self.log_noise.exp().diag())
                for i in range(1, max_photon)])
            
            #set_trace()
            
            return preprocRandomVariables.BatchRandomVariable(
                *[preprocRandomVariables.MixtureRandomVariableWithSampler(*p_PM_each, weights=photon_probs[:,i])
                  for i in range(photon_probs.size(1))]
                 )
            
        else: # We are just returning the predictive mean and variance
        
            # Get the input standard deviation
            latStd = getVar(latent_func).sqrt()


            photon_counts, photon_log_probs = (
                    self.getPhotonLogProbs(latent_func.mean().exp(), max_photon=float(max_photon), reNormalise = False))

            predVariancesPerPhoton = (photon_counts*(self.log_noise.exp())).squeeze()
            predVariancesPerPhoton[0] += self.log_noise_pedestal.exp().squeeze()

            # Get the photon probabilities for all latent inputs
            allPhotonLogProbs = torch.zeros([latent_func.mean().size(0), max_photon], device=input_device)


            # Store the resulting moments
            pred_moments = torch.zeros([latent_func.mean().size(0), 2], device=input_device)

            # Do minibatch integration to get expected probabilities by summing samples of log probabilties            
            for i in range(int(int(latStd.size(0))/batchsize)+1):
                cur_slice = slice(i*batchsize,min((i+1)*batchsize, latStd.size(0)))
                if cur_slice.start == cur_slice.stop:
                    break

                #set_trace()

                tmp=self.integrator.batch_integrate_gauss(
                    lambda x: self.getPhotonLogProbs(x.exp(), max_photon=float(max_photon), reNormalise = False)[1].permute(1,0,2),
                    mu = latent_func.mean()[cur_slice],
                    sig = latStd[cur_slice],
                    viewAs = [-1, 1, 1]
               )


                allPhotonLogProbs[cur_slice,:] = tmp.permute(1,0)

                #set_trace()

                # Set the predictive means
                pred_moments[cur_slice, 0] = allPhotonLogProbs[cur_slice,:].exp().matmul((
                    photon_counts*self.log_gain.exp()+self.offset).squeeze().unsqueeze(1)).squeeze()

                # Set the predictive variances
                #set_trace()

                # First weighted squared distance from global mean
                pred_moments[cur_slice, 1] = (allPhotonLogProbs[cur_slice,:].exp().mul(
                    ((photon_counts*self.log_gain.exp()+self.offset).view(1,-1) - pred_moments[cur_slice, 0].view(-1,1)).pow(2)
                        ).sum(1).squeeze())

                #set_trace()

                # Then the individual variances
                pred_moments[cur_slice, 1] += (allPhotonLogProbs[cur_slice,:].exp()
                                               .matmul(predVariancesPerPhoton.view(-1,1)).squeeze())

                # Once we have all probalities, estimate the output mixture variable means and variances

            return GaussianRandomVariable(pred_moments[:,0], gpytorch.lazy.DiagLazyVariable(pred_moments[:,1]))
        
        """
        # Create the mixture elements
        p_PM_each = [GaussianRandomVariable(self.offset, self.log_noise_pedestal.exp().diag())]
        p_PM_each.extend([
            GaussianRandomVariable(i*self.log_gain.exp()+self.offset, 
                                   i*self.log_noise.exp().diag())
            for i in range(1, max_photon)])

            def tmp_f(x): 
                tmp = MixtureRandomVariable(*p_PM_each, weights=x)
                return torch.stack([tmp.mean().data, tmp.var().data], dim=1).to(x.device)

            set_trace()

            pred_moments[cur_slice,:] = preprocUtils.apply(
                tmp_f,
                allPhotonLogProbs[cur_slice,:].exp(),
                dim = 0
            )
       """
        
        #set_trace()
        
        
###############################################################################################################        
###############################################################################################################
###############################################################################################################

        
        
def logistic(x,  x0=0., k=1., L=1.):
    return x.add(-x0).mul(-k).exp().add(1).reciprocal().mul(L)

def logit(p):
    return p.mul((1-p).reciprocal()).log()
        
#R. Dossi et al. / Nuclear Instruments and Methods in Physics Research A 451 (2000) 623}637
class PoissonInputUnderamplifiedPhotomultiplierLikelihood(BasePhotomultiplierLikelihood):
    """
    Assumes that the incoming discrete number gets linearly multiplied by log_gain.exp(),
    and the observed value is a Normal distribution with a given input-offset, 
    whose mean and variance comes from the observed count.
    
    There is also an explicit "pedestal", which models the distribution around 0 counts.
    
    Furthermore the Single PhotoElectron Response (SER) has an underamplified portion described as an exponential.
    For f1 >= 2, the multiple photoelectron response is approximated as a gaussian with the linearly scaled
        mean and variance of the SER.
    
    p(f1 | f) = Poisson(f)
    p(y -offset | f1) = 
        {
            N_y (  0 , sigma_y_0)                       if f1 == 0
            pE * Exp_y(A) + (1-pE)*N_y (gain, s0^2 )    if f1 == 1
            N_y ( f1*x1,  f1^2 * s1^2 )                 if f1 >= 2,
                
                    where we x1 and s1 are the moments of p(y-offset|f1==1):
                        x1 \approx (1-pE)*gain + pe*A
                        s1 \approx (1-pE)*(gain^2+s0^2) + pe*2*A^2-x1^2
    
        # noise = s0^2
        # noise_pedestal = sigma_y_0
        # nderamplified_probability = pE
        # underamplified_amplitude = A
    """
    def __init__(self, gain=None, offset=None, noise=None, noise_pedestal=None, 
                 underamplified_probability=None, underamplified_amplitude=None): 
        super(PoissonInputUnderamplifiedPhotomultiplierLikelihood, self).__init__(
            gain = gain, offset = offset, noise=noise)
        
        self.register_parameter(name="log_noise_pedestal", 
                                parameter=toTorchParam(noise_pedestal if noise_pedestal is not None else 1e-2, ndims=1, to_log=True), 
                                prior=SmoothedBoxPrior(-10, 5, sigma = 0.1))
        
        self.register_parameter(name="logit_underamplified_probability", 
                                parameter=toTorchParam(
                                    logit(underamplified_probability)
                                        if underamplified_probability is not None 
                                        else -2., ndims=1), 
                                prior=SmoothedBoxPrior(-6, 0, sigma = 0.01)) # between 0 and 0.5
        
        self.register_parameter(name="log_underamplified_amplitude", 
                                parameter=toTorchParam(
                                    underamplified_amplitude 
                                        if underamplified_amplitude is not None 
                                        else self.log_gain.clone().exp().div(2), ndims=1, to_log=True), 
                                prior=SmoothedBoxPrior(-8, 5, sigma = 0.1))
    
    
    def getLogProbSumOverTargetSamples(self, p_PM, cur_target_slice):
            """ 
            As we are supposedly dealing with truncated normals and expmod-normals, we need to replace
            log probabilities of 0s in the cur_target with the (log) CDF at 0 instead of the log_prob
            
            p_PM is going to be a list here, with:
              p_PM[0] = noise distribution
              p_PM[1] = Single photon (underamplified) distribution
              p_PM[2] = Multi-photon distribution with mean and var linearly amplified from SER
            
            """
            

            #set_trace()
            
            allLogProbs = torch.cat(
                [p_PM[0].log_prob(cur_target_slice.view(-1,1)).view(-1,1),
                p_PM[1].log_prob(cur_target_slice.view(-1,1)).view(-1,1),
                p_PM[2].log_prob(cur_target_slice.view(-1,1))],
                dim = 1)
            
            #set_trace()
            
            # Correct for the less than 1 observations with log CDF instead of log_prob
            if (cur_target_slice<=0.).sum()>0:       
                allLogProbs[cur_target_slice<=0., :] = torch.cat(
                    [p_PM[0].cdf(0.).log().view(-1),
                    p_PM[1].cdf(0.).log().view(-1),
                    p_PM[2].cdf(0.).log().view(-1)],
                    dim = 0)
                
            
            # Sum over target samples
            return allLogProbs.unsqueeze(1).sum(-2).squeeze() # This should work for 1d or 2d targets
    
        
    def createResponseDistributions(self, photon_counts):
               # p_PM is going to be a list here, with:
        #   p_PM[0] = noise distribution
        #   p_PM[1] = Single photon (underamplified) distribution
        #   p_PM[2] = Multi-photon distribution with mean and var linearly amplified from SER
        
        pE = logistic(self.logit_underamplified_probability).squeeze()
        A = self.log_underamplified_amplitude.exp().squeeze()
        gain = self.log_gain.exp().squeeze()
        s0 = self.log_noise.mul(0.5).exp()
        
        
        p_PM = list()
        p_PM.append(Normal(loc=self.offset.squeeze(), 
                                               scale = self.log_noise_pedestal.mul(0.5).exp().squeeze()))
        
        p_PM.append(SingleElectronResponse(loc=(gain+self.offset).squeeze(), 
                                           scale=s0.squeeze(),
                                           pedestal_loc = self.offset.squeeze(), 
                                           pedestal_scale = self.log_noise_pedestal.mul(0.5).exp().squeeze(),
                                           underamplified_amplitude = A,
                                           underamplified_probability = pE
                                          )
                   )
                    
        
        # For higher photon counts we ignore convolution with the noise and compute moments of idealised SER0
        multiphoton_loc_base = pE*A + (1-pE)*gain                         
        multiphoton_scale_base = (pE*2.*A.pow(2.) 
                                  + (1.-pE)*(gain.pow(2.)+s0.pow(2.)) 
                                  - multiphoton_loc_base.pow(2.)
                                 ).sqrt() # second moment of Exp and Normal - mean^2
        
        p_PM.append(Normal(loc=(photon_counts[2:]*multiphoton_loc_base+self.offset).squeeze(), 
                                               scale=(photon_counts[2:]*multiphoton_scale_base).squeeze())
                                         )
        
        return p_PM
        
    
    def single_log_prob(self, inp, cur_target, batchsize = int(500), max_photon=15.):
        """
        Input is directly from latent_func, means it still needs to be exponentiated before multiplying.
        Input is the log mean of a Poisson distribution
        
        Input shape is [QuadratureWeights, N]
        cur_target is [N,]
        """
        
        # Have to do this in mini-batches as the resulting QuadWeights x MaxPhotons x N array is too big
        all_res = torch.zeros_like(inp)
        
        photon_counts, photon_log_probs = (
                self.getPhotonLogProbs(inp[:,:2].exp(), max_photon=float(max_photon), reNormalise = False))

        
        p_PM = self.createResponseDistributions(photon_counts)
 
        #set_trace()
        
        
        for i in range(int(int(inp.size(1))/batchsize)+1):
            cur_slice = slice(i*batchsize,min((i+1)*batchsize, inp.size(1)))
            if cur_slice.start == cur_slice.stop:
                break
                
            inp_cur = inp[:,cur_slice]
            if inp_cur.ndimension()==1:
                inp_cur = inp_cur.unsqueeze(1)
            
            photon_counts, photon_log_probs = (
                self.getPhotonLogProbs(inp_cur.exp(), max_photon=float(max_photon), reNormalise = False))

            ### return photon_counts, photon_log_probs, p_PM

            log_prob_sum_over_target_samples = self.getLogProbSumOverTargetSamples(p_PM, cur_target[cur_slice])
            if log_prob_sum_over_target_samples.ndimension()==1: # Correct for if we only get a single input value
                log_prob_sum_over_target_samples = log_prob_sum_over_target_samples.unsqueeze(0)



            all_res[:,cur_slice] += (log_prob_sum_over_target_samples.permute(1,0).unsqueeze(1) + photon_log_probs).logsumexp(dim=0)
            
            #del log_prob_sum_over_target_samples, photon_counts, photon_log_probs, p_PM
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return all_res # Same dimensionality as inp [QuadratureWeights, N]
    
    
    
    def forward(self, latent_func, approx=False, max_photon=int(20), batchsize = int(1000)):
        """
        Computes predictive distributions p(y|x) given a latent distribution
        p(f|x). To do this, we solve the integral:
            p(y|x) = \int p(y|f)p(f|x) df
            
        As the true representation gets storage expensive for large latent_func dimensionalities, 
        we provide an approximate method by default in which we only store the true mean and variance for each output
        """
        
        
        input_device = latent_func.mean().device
        
        if approx is False: ## Return the full Batch of MixtureRandomVariables (huge object, use only for few predictions)
            
            # Get the input standard deviation
            latStd = getVar(latent_func).sqrt()
            
            # Get the photon log probabilities
            photon_log_probs = self.integrator.batch_integrate_gauss(
                    lambda x: self.getPhotonLogProbs(x.exp(), max_photon=float(max_photon), reNormalise = False)[1].permute(1,0,2),
                    mu = latent_func.mean(),
                    sig = latStd,
                    viewAs = [-1, 1, 1]
               )
            
            
            
            # As these probabilities are a result of non-perfect integration, we need to renormalise them outside of the integration
            photon_log_probs -= photon_log_probs.logsumexp(0).view(1,-1)
            
            # For very low probabilities, the renomalisation may not have been perfect, renormalise once more in probability space
            photon_probs = photon_log_probs.exp()
            photon_probs = photon_probs.div(photon_probs.sum(0).unsqueeze(0))
            
            pE = logistic(self.logit_underamplified_probability).squeeze()
            A = self.log_underamplified_amplitude.exp().squeeze()
            gain = self.log_gain.exp().squeeze()
            s0 = self.log_noise.mul(0.5).exp()
            
            # Underlying mixture elements
            p_PM_each = [GaussianRandomVariable(self.offset, self.log_noise_pedestal.exp().diag())]
            
            
            
            p_PM_each.append(SingleElectronResponseRandomVariable(loc=(gain+self.offset).squeeze(), 
                                           scale=s0.squeeze(),
                                           pedestal_loc = self.offset.squeeze(), 
                                           pedestal_scale = self.log_noise_pedestal.mul(0.5).exp().squeeze(),
                                           underamplified_amplitude = A,
                                           underamplified_probability = pE
                                          )
                   )
                    
        
            # For higher photon counts we ignore convolution with the noise and compute moments of idealised SER0
            multiphoton_loc_base = pE*A + (1-pE)*gain                         
            multiphoton_scale_base = (pE*2.*A.pow(2.) + (1.-pE)*(gain.pow(2)+s0.pow(2)) - multiphoton_loc_base.pow(2.)).sqrt() # second moment of Exp and Normal - mean^2
            
            p_PM_each.extend([
                GaussianRandomVariable((i*multiphoton_loc_base+self.offset).view(-1), 
                                   (i*multiphoton_scale_base.view(1,1)).pow(2.))
                for i in range(2, max_photon)])
            
            
            
            return preprocRandomVariables.BatchRandomVariable(
                *[preprocRandomVariables.MixtureRandomVariableWithSampler(*p_PM_each, weights=photon_probs[:,i])
                  for i in range(photon_probs.size(1))]
                 )
            
        else: # We are just returning the predictive mean and variance
        
            raise NotImplementedError
    
