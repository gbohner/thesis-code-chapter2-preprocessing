import torch
import math
import gpytorch
import gpytorch.models
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.random_variables import GaussianRandomVariable

import torch
from torch.autograd import Variable
from gpytorch.kernels import GridInterpolationKernel, Kernel
from gpytorch.lazy import LazyVariable, LazyEvaluatedKernelVariable
from gpytorch.lazy import DiagLazyVariable, InterpolatedLazyVariable, AddedDiagLazyVariable
from gpytorch.random_variables import GaussianRandomVariable
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import MVNVariationalStrategy
from gpytorch.models.abstract_variational_gp import AbstractVariationalGP
from gpytorch import beta_features
from gpytorch.utils import Interpolation, left_interp


from preprocUtils import *
from preprocRandomVariables import *
import preprocInterpolation

from IPython.core.debugger import set_trace


class MeanVariationalStrategy(gpytorch.variational.VariationalStrategy):
    """
    Assumes that the variational covar is the same as the prior one, and only deals with the means
    This way the prior covariance function affects the variational mean in expected ways
    
    When variational covar == prior covar, we only need to compute the prior quadratic form
    (u_var - u_prior).T * prior_covar.inv() * (u_var - u_prior)
    """
    def kl_divergence(self):
        prior_mean = self.prior_dist.mean()
        prior_covar = self.prior_dist.covar()
        if not isinstance(prior_covar, LazyVariable):
            prior_covar = NonLazyVariable(prior_covar)
        prior_covar = prior_covar.add_jitter()

        variational_mean = self.variational_dist.mean() 
        
        mean_diffs = prior_mean - variational_mean
        
        #set_trace()
        
        inv_quad_form = prior_covar.inv_quad_log_det(
            inv_quad_rhs=mean_diffs.unsqueeze(-1), 
            log_det=False
        )
        
        res = 0.5*inv_quad_form[0]
        
        return res


class NoCovarAbstractVariationalGP(AbstractVariationalGP):
    def __init__(self, inducing_points):
        super(AbstractVariationalGP, self).__init__()
        if not torch.is_tensor(inducing_points):
            raise RuntimeError("inducing_points must be a Tensor")
        n_inducing = inducing_points.size(0)
        self.register_buffer("inducing_points", inducing_points)
        self.register_buffer("variational_params_initialized", torch.zeros(1))
        self.register_parameter(name="variational_mean", parameter=torch.nn.Parameter(torch.zeros(n_inducing)))
        self.register_variational_strategy("inducing_point_strategy")
    
    
class BaseGridInterpolationModel(NoCovarAbstractVariationalGP):
    """
    Model that smartly initialises it's own parameter in the init, given sufficient inputs
    
    It is built for fast variational inference (arbitrary likelihoods) and 
    uses interpolation between the inducing points for fast predictions
    """
    def __init__(self, train_x, grid_size = 50, grid_bounds=None, force_prior_covar = False, 
                 interp_point_number = int(7), **kwargs):
        
        #### --------------------------
        #### Set up inducing point grid
        #### --------------------------
        
        
        # Set grid bounds
        if grid_bounds is None:
            dataStats = getDataStatistics(train_x)
            grid_bounds = dataStats['x_minmax'].clone()
            grid_bounds[0,:] -= dataStats['x_width']/(grid_size-1)*2
            grid_bounds[1,:] += dataStats['x_width']/(grid_size-1)*2
            #grid_bounds[0,:] -= dataStats['x_width']/(grid_size-(interp_point_number+1))*((interp_point_number+1)/2)
            #grid_bounds[1,:] += dataStats['x_width']/(grid_size-(interp_point_number+1))*((interp_point_number+1)/2)
            grid_bounds = grid_bounds.unbind(1)
        
        # THE CURRENT INTERPOLATION SYSTEM HAS VERY VERY BAD BOUNDARY BEHAVIOR
        # IN ORDER TO AVOID THAT, WE NEED AT LEAST interp_point_number+1)/2 GRID LOCATIONS OUTSIDE OF THE ACTUAL EXPECTED AREA 
            # (thus the -(interp_point_number+1))*((interp_point_number+1)/2) )
        
        
        # Create grid
        grid = torch.cat([
            torch.linspace(grid_bounds[i][0], grid_bounds[i][1], grid_size).unsqueeze(0)
            for i in range(len(grid_bounds))
            ], 
          dim=0)
      
        inducing_points = torch.zeros(int(pow(grid_size, len(grid_bounds))), len(grid_bounds))
        prev_points = None
        for i in range(len(grid_bounds)):
            for j in range(grid_size):
                inducing_points[j * grid_size ** i : (j + 1) * grid_size ** i, i].fill_(grid[i, j])
                if prev_points is not None:
                    inducing_points[j * grid_size ** i : (j + 1) * grid_size ** i, :i].copy_(prev_points)
            prev_points = inducing_points[: grid_size ** (i + 1), : (i + 1)]
        
        #self.register_buffer("inducing_points", inducing_points)
        
        
        #### ---------------------------------
        #### Initialise variational parameters
        #### ---------------------------------
        
        # Initialise the module collector AbstractVariationalGP
            # this provides prior_output() and variational_output()
        
        if force_prior_covar: # Do not set up chol_variational_covar
            super(BaseGridInterpolationModel, self).__init__(inducing_points) 
        else: #set up chol_variational_covar
             super(NoCovarAbstractVariationalGP, self).__init__(inducing_points)
        
        self.register_buffer("grid", grid)
        self.register_buffer("interp_points", 
                             torch.tensor(range(int(-(interp_point_number-1)/2), int((interp_point_number+1)/2 +1))))
        self.force_prior_covar = force_prior_covar
        
        #### ---------------------------------
        #### Speed up training by pre-computing the kernel between training and inducing points
        #### ---------------------------------

        # Save buffer for training:
        self.register_buffer("training_inputs", train_x)
        self.has_training_cache = False
    
    
    def prior_output(self):
        # Overwrite this, because "res.covar.evaluate_kernel()" fails for some reason, just use normal evaluate
        res = super(AbstractVariationalGP, self).__call__(Variable(self.inducing_points))
        if not isinstance(res, GaussianRandomVariable):
            raise RuntimeError("%s.forward must return a GaussianRandomVariable" % self.__class__.__name__)

        res = GaussianRandomVariable(res.mean(), res.covar().evaluate())
        return res
    
#    def variational_output(self): 
#        # Overwrite to detach chol_variational_covar somehow?
#        return GaussianRandomVariable(self.variational_mean,
#                                     torch.eye(len(self.variational_mean)).type_as(self.variational_mean)
#                                     )
    
        
    def _initalize_variational_parameters(self, prior_output):
        mean_init = prior_output.mean().data
        mean_init += mean_init.new(mean_init.size()).normal_().mul_(1e-1)
        chol_covar_init = torch.eye(len(mean_init)).type_as(mean_init)
        chol_covar_init += chol_covar_init.new(chol_covar_init.size()).normal_().mul_(1e-1)
        #set_trace()
        
        # # Set chol_var_cov as prior_cov.root_decomp
        #self.chol_variational_covar.data = 1e-3*self.chol_variational_covar.data
        #chol_covar_init = gpytorch.utils.pivoted_cholesky.pivoted_cholesky(
        #    prior_output.covar(), max_iter=300).transpose(0,1)#.evaluate()
        #self.chol_variational_covar.data[:,:chol_covar_init.size(1)].copy_(chol_covar_init)
        
        
        self.variational_mean.data.copy_(mean_init)
        if not self.force_prior_covar:
            self.chol_variational_covar.data.copy_(chol_covar_init)

    # Get the interpolation from the variational parameters to arbitrary inputs
    def _compute_grid(self, inputs):
        if inputs.ndimension() == 1:
            inputs = inputs.unsqueeze(1)

        #interp_indices, interp_values = Interpolation().interpolate(Variable(self.grid), inputs, interp_points=range(-2,2)) # TODO - Only -2,2 works, write a better interpolation
        
        interp_indices, interp_values = preprocInterpolation.RBFInterpolation().interpolate(
            Variable(self.grid), inputs, interp_points=list(self.interp_points.unbind()))
        
        return interp_indices, interp_values


    def train(self, mode=True):
        # Delete the cache on triggering train(), either true of false (saves memory)
        if self.has_training_cache:
            del self._cached_interp_indices
            del self._cached_interp_values
            self.has_training_cache = False

        return super(BaseGridInterpolationModel, self).train(mode)


    # This seems unused:
    #def _inducing_forward(self):
    #    inducing_points_var = Variable(self.inducing_points)
    #    return super(BaseGridInterpolationModel, self).forward(inducing_points_var, inducing_points_var)

    #### ---------------------------------
    #### Define the main __call__ function
    #### ---------------------------------

    def __call__(self, inputs, **kwargs):
        # Training mode: optimizing
        if self.training:
            if not torch.equal(inputs.data, self.training_inputs):
                # Recompute interpolation grid
                interp_indices, interp_values = self._compute_grid(inputs)
                self._cached_interp_indices = interp_indices
                self._cached_interp_values = interp_values    
                self.has_training_cache = True
                #raise RuntimeError("You must train on the training inputs!")

        if self.training or beta_features.diagonal_correction.on():
            prior_output = self.prior_output()
            # Initialize variational parameters, if necessary
            if not self.variational_params_initialized[0]:
                self._initalize_variational_parameters(prior_output)
                self.variational_params_initialized.fill_(1)

        # Variational output
        variational_output = self.variational_output()

        # Update the variational distribution
        if self.training:
            if not self.force_prior_covar:
                #set_trace()
                new_variational_strategy = MVNVariationalStrategy(variational_output, prior_output)
            else:
                new_variational_strategy = MeanVariationalStrategy(variational_output, prior_output)
            self.update_variational_strategy("inducing_point_strategy", new_variational_strategy)

        # Get interpolations
        if self.training:
            if not self.has_training_cache:
                interp_indices, interp_values = self._compute_grid(inputs)
                self._cached_interp_indices = interp_indices
                self._cached_interp_values = interp_values    
                self.has_training_cache = True
            else:
                interp_indices = self._cached_interp_indices
                interp_values = self._cached_interp_values
        else:
            interp_indices, interp_values = self._compute_grid(inputs)

        #set_trace()

        # Compute test mean
        # Left multiply samples by interpolation matrix
        test_mean = left_interp(interp_indices, interp_values, variational_output.mean().unsqueeze(-1))
        test_mean = test_mean.squeeze(-1)

        # Compute test covar
        if not self.force_prior_covar:
            #set_trace()
            test_covar = InterpolatedLazyVariable(
                variational_output.covar(), interp_indices, interp_values, interp_indices, interp_values
            )
        else:
            if not self.training:
                prior_output = self.prior_output()
            test_covar = InterpolatedLazyVariable(
                prior_output.covar(), interp_indices, interp_values, interp_indices, interp_values
            )

        # Diagonal correction
        if gpytorch.beta_features.diagonal_correction.on():
            #set_trace()
            from gpytorch.lazy import AddedDiagLazyVariable

            prior_covar = InterpolatedLazyVariable(
                prior_output.covar(), interp_indices, interp_values, interp_indices, interp_values
            )
            diagonal_correction = DiagLazyVariable((self.covar_diag(inputs) - prior_covar.diag()) * 0)
            test_covar = AddedDiagLazyVariable(test_covar, diagonal_correction)

        output = GaussianRandomVariable(test_mean, test_covar)

        return output





class FlexibleVariationalGridInterpolationModel(BaseGridInterpolationModel):
    """
    Model that smartly initialises it's own parameter in the init, given sufficient inputs
    
    It is built for fast variational inference (arbitrary likelihoods) and a grid-interpolation kernel
    """
    def __init__(self, train_x, train_y, 
                 mean_module=gpytorch.means.ZeroMean(), covar_module=RBFKernel(), 
                 grid_size = 50, grid_bounds=None, **kwargs
                ):
        
        # Initialise the grid inducing variational GP
        super(FlexibleVariationalGridInterpolationModel, self).__init__(train_x, grid_size, grid_bounds, **kwargs)
        
        # Add the mean and covariance modules
        self.mean_module = mean_module
        self.covar_module = covar_module
        
    
    def forward(self, x):
        #set_trace()
        # ZeroMean needs to be summed to get rid of wrong dimensionality but ConstantMean does not
        mean_x = self.mean_module(x).unsqueeze(-1).sum(1).squeeze() 
        covar_x = self.covar_module(x)
        return GaussianRandomVariable(mean_x, covar_x)









class preprocModel2DGridTwoRBFKernels(gpytorch.models.GridInducingVariationalGP):
    def __init__(self):
        """
        ..note
            # Change super to work with reloading, detailed in a blog-post here (seems it is not needed with direct import, 
            # only with from module import *):
            # https://thingspython.wordpress.com/2010/09/27/another-super-wrinkle-raising-typeerror/
            #self.as_super = super(preprocModel2DGridTwoRBFKernels, self)
            #self.as_super.__init__()
        """
        super(preprocModel2DGridTwoRBFKernels, self).__init__(
            grid_size=30, grid_bounds=[(-0.05, 1.05), (-0.05, 1.05)])
        self.mean_module = ConstantMean(prior=SmoothedBoxPrior(-10, 10))
        self.covar_module = RBFKernel(
            log_lengthscale_prior=SmoothedBoxPrior(math.exp(-5), math.exp(2), sigma=0.1, log_transform=True)
        ) + RBFKernel(
            log_lengthscale_prior=SmoothedBoxPrior(math.exp(-5), math.exp(2), sigma=0.1, log_transform=True)
        )
        self.register_parameter(
            name="log_outputscale",
            parameter=torch.nn.Parameter(torch.tensor([0.])),
            prior=SmoothedBoxPrior(math.exp(-5), math.exp(3), sigma=0.1, log_transform=True),
        )



    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x) * self.log_outputscale.exp()
        return GaussianRandomVariable(mean_x, covar_x)    
 

# ### 1D model w/ inducing points -------------------------
# class GPRegressionModel(gpytorch.models.GridInducingVariationalGP):
#     def __init__(self):        
#         super(GPRegressionModel, self).__init__(grid_size=20, grid_bounds=[(-0.05, 1.05)])
# #         super(GPRegressionModel, self).__init__(grid_size=20, 
# #                                                 grid_bounds=[(-0.05, 1.05), (-0.05, 1.05)])
#         self.mean_module = ConstantMean(prior=SmoothedBoxPrior(-10, 100, log_transform=True))
#         self.covar_module = RBFKernel(
#             log_lengthscale_prior=SmoothedBoxPrior(math.exp(-3), math.exp(6), sigma=0.1, log_transform=True)
#         )
#         self.register_parameter(
#             name="log_outputscale",
#             parameter=torch.nn.Parameter(torch.tensor([1])),
#             prior=SmoothedBoxPrior(math.exp(-5), math.exp(6), sigma=0.1, log_transform=True),
#         )



# ## 2D model with kernel interpolation -------------------
# class GPRegressionModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = ConstantMean(prior=SmoothedBoxPrior(-10, 100, sigma=1., log_transform=True))
#         self.covar_module = RBFKernel()
#         self.register_parameter(
#             name="log_outputscale",
#             parameter=torch.nn.Parameter(torch.tensor([1])),
#             prior=SmoothedBoxPrior(math.exp(-3), math.exp(3), sigma=1., log_transform=True),
#         )
#         self.base_covar_module = RBFKernel()
#         self.covar_module = GridInterpolationKernel(self.base_covar_module, grid_size=30,
#                                                     grid_bounds=[(0, 1), (0, 1)])
#         self.register_parameter(
#             name="log_outputscale",
#             parameter=torch.nn.Parameter(torch.tensor([1])),
#             prior=SmoothedBoxPrior(math.exp(-5), math.exp(6), sigma=0.1, log_transform=True),
#         )