import torch
import gpytorch
import math
from numpy import polynomial
# Implement as in https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss%E2%80%93Legendre_quadrature
class QuadratureIntegratorTorch(gpytorch.Module):
    """
        Computes the quadrature samples and weights once for a given degree and stores them for reuse
        Then provides 1 dimensional integrals:
            integrate(f, a, b) = \int_a^b f(x) dx 
            integrate_discrete(self, f, a=0.0, b=10.0) = \sum_x=a^b f(x)
            
        TODO: provide other kinds of integration where the function is weighted 
                (such as Gauss-Hermite or Gauss_Laguerre)
                
        TODO: Estimate error of the integrator and add more points if needed 
            (Gauss-Kronrod adds n+1 points, and one can do that iteratively until estimated error is below some threshold)
    """
    def __init__(self, deg):
        super(QuadratureIntegratorTorch, self).__init__()
        self.register_buffer("deg", torch.tensor(deg, dtype=torch.int))
        
        # Gauss-Legendre integrator (equal weights over the range)
        s, w  = polynomial.legendre.leggauss(int(self.deg))
        self.register_buffer("gl_s", torch.tensor(s).float())
        self.register_buffer("gl_w", torch.tensor(w).float())
        
        s, w = polynomial.hermite.hermgauss(int(self.deg))
        self.register_buffer("gh_s", torch.tensor(s).float())
        self.register_buffer("gh_w", torch.tensor(w).float())
    
    def change_range(self, x, a=-1.0, b=1.0):
        return 0.5*((b-a)*x + (b+a))
    
    def integrate(self, f, a=-1.0,b=1.0):
        x = self.change_range(self.gl_s, a=a, b=b)
        return (b-a)/2.0*torch.sum(self.gl_w*f(x))
    
    def batch_integrate(self, f, a=-1.0,b=1.0, viewAs=None):
        x = self.change_range(self.gl_s, a=a, b=b).unsqueeze(-1)
        
        viewAs = viewAs if viewAs is not None else [-1, 1]
        
        return ((b-a)/2.0*
                self.gl_w.view(*viewAs)
                .mul(f(x))
                .sum(0)
               )
    
    def integrate_gauss(self, f, mu=0.0, sig=1.0, a=-1.0, b=1.0):
        # Change of variables for samples for mu and sigma:
        x = self.gh_s.mul(torch.tensor(2., device=self.deg.device).sqrt()).mul(sig) + mu
        
        # Change of range for a and b
        x = self.change_range(x, a=a, b=b)
        
        return torch.tensor(1./math.pi, device=self.deg.device).sqrt() * torch.sum(self.gh_w * f(x))
    
    def batch_integrate_gauss(self, f, mu, sig, viewAs=None):
        """
        Carries out integration for vectors mu and sigma
        
        Expands the weights to [n_weights, 1] (expecting the f(x) to be [n_weights, N]
         OR sets them to weights.view(*viewAs)
        """
        x = (self.gh_s
             .unsqueeze(-1)
             .expand(-1, mu.size(0))
             .mul(torch.tensor(2., device=self.deg.device).sqrt())
             .mul(sig.view(1,-1))
             + mu.view(1,-1)
            )
        
        viewAs = viewAs if viewAs is not None else [-1, 1]
        
        return (self.gh_w.view(*viewAs)
                .mul(f(x))
                .mul(torch.tensor(1./math.pi, device=self.deg.device).sqrt())
                .sum(0)
                )
        
    def integrate_discrete(self, f, a=0.0, b=10.0, step=1.0):
        return torch.sum([f(x) for x in torch.arange(a, b+step, step)])