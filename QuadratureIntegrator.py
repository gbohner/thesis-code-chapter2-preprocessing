import autograd.numpy as np
from numpy import polynomial
# Implement as in https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss%E2%80%93Legendre_quadrature
class QuadratureIntegrator(object):
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
        self.deg = deg
        self.gl = dict()
        self.gl['samples'], self.gl['weights'] = polynomial.legendre.leggauss(deg)
        
    def integrate(self, f, a=-1.0,b=1.0):
        x = 0.5*((b-a)*self.gl['samples'] + (b+a))
        return (b-a)/2.0*np.sum(self.gl['weights']*f(x))
    
    def integrate_discrete(self, f, a=0.0, b=10.0, step=1.0):
        return np.sum([f(x) for x in np.arange(a, b+step, step)])