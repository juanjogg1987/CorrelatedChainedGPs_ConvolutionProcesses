# Code of Zero-Inflated Poisson Likelihood develped by Juan-Jose Giraldo
# based on the previous likelihood codes for the HetMOGP model by Pablo Moreno-Munoz (Universidad Carlos III)
# University of Sheffield

import sys
import numpy as np
from GPy.likelihoods import link_functions
from GPy.likelihoods import Likelihood
from GPy.util.misc import safe_exp, safe_square
from scipy.special import beta, betaln, psi, zeta, gamma, gammaln
from scipy import special
from functools import reduce
from scipy.special import logsumexp

class ZIPoisson(Likelihood):
    """
    Zero-Inflated (ZI) Poisson likelihood with a latent function over its parameters

    """

    def __init__(self, gp_link=None):
        if gp_link is None:
            gp_link = link_functions.Identity()

        super(ZIPoisson, self).__init__(gp_link, name='ZIPoisson')

    def pdf(self,F, y, Y_metadata=None):
        #Here f[:,0] is latent GP over phi = sigmoid(f[:,0]) and f[:,1] is latent GP over lamb_poisson = exp(f[:,1])
        pdf = safe_exp(self.logpdf(F, y, Y_metadata=None))
        return pdf

    def logpdf_poisson(self,f, y, Y_metadata=None):
        ef = safe_exp(f[:,None])
        logpdf_poiss = -ef + (y*f[:,None]) - special.gammaln(y+1)
        return logpdf_poiss


    def log_ZI_argument(self,y, F):
        # Computing the Zero-Inflated (ZI) part of the likelihood: \log (phi*exp(lamb_poisson)+1-phi)
        # Here f[:,0] is latent GP for phi = sigmoid(f[:,0]) and f[:,1] is latent GP for lamb_poisson = exp(f[:,1])
        ef0 = safe_exp(F[:, 0,None])
        phi = ef0 / (1 + ef0)
        phi = np.clip(phi, 1.0e-15, 1.0)  # numerical stability
        res = 1 - phi  # Here phi should be a vector of probabilities, i.e. a squashing function of the latent GP
        lamb_poisson = safe_exp(F[:, 1,None])
        lamb_poisson = np.clip(lamb_poisson, 0.0, 1.797e308)  # numerical stability
        # res = y.copy()
        whichzeros = (y == 0)
        res[whichzeros] = res[whichzeros] + phi[whichzeros] * safe_exp(lamb_poisson[whichzeros])
        return np.log(res)

    def logpdf(self,F, y, Y_metadata=None):
        logpdf = self.log_ZI_argument(y,F) + self.logpdf_poisson(F[:,1], y, Y_metadata)
        return logpdf

    def logpdf_sampling(self, F, y, Y_metadata=None):
        logpdf = np.zeros((y.shape[0],F.shape[2]))
        for i in range(F.shape[2]):
            logpdf[:,i] = self.logpdf(F[:,:,i], y, Y_metadata=None)[:,0]

        return logpdf

    def samples(self, F ,num_samples, Y_metadata=None):
        ef0 = safe_exp(F[:, 0,None])
        phi = ef0 / (1 + ef0)
        phi = np.clip(phi, 1.0e-15, 1.0)  # numerical stability
        lamb_poisson = safe_exp(F[:, 1,None])
        lamb_poisson = np.clip(lamb_poisson, 0.0, 1.797e308)  # numerical stability
        probability_for_zero = phi #+ (1-phi)*safe_exp(-lamb_poisson)
        #print('f and phi:',f[:,0],phi)
        bernoulli_samples = np.random.binomial(n=1,p=1-probability_for_zero)
        poisson_samples = np.random.poisson(lam=lamb_poisson)
        samples = bernoulli_samples*poisson_samples
        return samples

    def dlogp_df(self,F, y, Y_metadata=None):
        ef0 = safe_exp(F[:, 0,None])
        phi = ef0 / (1 + ef0)
        phi = np.clip(phi, 1.0e-15, 1.0)  # numerical stability
        lamb_poisson = safe_exp(F[:, 1,None])
        lamb_poisson = np.clip(lamb_poisson, 1.0e-9, 1.797e308)  # numerical stability
        dlogp_f0 = -phi

        whichzeros = (y == 0)
        lim_max = 1.797e308
        #phi_times_exp = phi[whichzeros] * safe_exp(lamb_poisson[whichzeros])
        #print('phi_times_exp:',phi_times_exp)
        #phi_times_exp_1_phi = np.clip( (phi_times_exp + 1 - phi[whichzeros]),-lim_max,lim_max)
        dlogp_f0[whichzeros] = (1-safe_exp(-lamb_poisson[whichzeros]))*(1-phi[whichzeros])*phi[whichzeros]/(phi[whichzeros]+safe_exp(-lamb_poisson[whichzeros])*(1-phi[whichzeros]))

        #dlogp_f0[whichzeros] = safe_exp( np.log((safe_exp(lamb_poisson[whichzeros])-1)*(phi[whichzeros]*(1-phi[whichzeros])))- np.log(phi_times_exp_1_phi))

        dlogp_f1 = - lamb_poisson + y
        #dlogp_f1[whichzeros] = dlogp_f1[whichzeros] + safe_exp( np.log(phi_times_exp*lamb_poisson[whichzeros]) - np.log(phi_times_exp_1_phi))
        dlogp_f1[whichzeros] = dlogp_f1[whichzeros] + phi[whichzeros]*lamb_poisson[whichzeros]/(phi[whichzeros]+safe_exp(-lamb_poisson[whichzeros])*(1-phi[whichzeros]))

        if np.isinf(dlogp_f0).sum()>0:
            print('inf first derivative f0')
            whichinf = np.isinf(dlogp_f0)
            getsign = np.sign(dlogp_f0[whichinf])
            dlogp_f0[whichinf] = lim_max
            dlogp_f0[whichinf] = dlogp_f0[whichinf] * getsign
        if np.isinf(dlogp_f1).sum()>0:
            print('inf first derivative f1')
            whichinf = np.isinf(dlogp_f1)
            getsign = np.sign(dlogp_f1[whichinf])
            dlogp_f1[whichinf] = lim_max
            dlogp_f1[whichinf] = dlogp_f1[whichinf] * getsign

        return dlogp_f0 , dlogp_f1

    def d2logp_df2(self,F, y, Y_metadata=None):
        ef0 = safe_exp(F[:, 0,None])
        phi = ef0 / (1 + ef0)
        phi = np.clip(phi, 1.0e-15, 1.0)  # numerical stability
        lamb_poisson = safe_exp(F[:, 1,None])
        lamb_poisson = np.clip(lamb_poisson, 0.0, 1.797e308)  # numerical stability
        dphi_df0 = phi*(1-phi)
        d2logp_f0 = -2*phi**4+3*phi**3-2*phi**2
        d2logp_f1 = - lamb_poisson
        #print('hola mundo')
        lim_max = 1.797e308
        whichzeros = (y == 0)

        exp_neg_lamb = safe_exp(-lamb_poisson[whichzeros])
        d2logp_f0[whichzeros] = (dphi_df0[whichzeros]**2)*((1-exp_neg_lamb)*(1-2*phi[whichzeros])*(phi[whichzeros]+exp_neg_lamb*(1-phi[whichzeros]))-(1-exp_neg_lamb)**2)/((phi[whichzeros]+exp_neg_lamb*(1-phi[whichzeros]))**2)
        exp_1_phi = exp_neg_lamb * (1 - phi[whichzeros])
        d2logp_f1[whichzeros] = d2logp_f1[whichzeros] + ((phi[whichzeros]*lamb_poisson[whichzeros])*(phi[whichzeros]+exp_1_phi)+(phi[whichzeros]*lamb_poisson[whichzeros]**2)*exp_1_phi)/((phi[whichzeros]+exp_1_phi)**2)

        if np.isnan(d2logp_f0).sum() > 0 or np.isnan(d2logp_f1).sum()>0:
            print('nan')

        if np.isinf(d2logp_f0).sum()>0:
            print('inf sec derivative f0')
            whichinf = np.isinf(d2logp_f0)
            getsign = np.sign(d2logp_f0[whichinf])
            d2logp_f0[whichinf] = lim_max
            d2logp_f0[whichinf] = d2logp_f0[whichinf]*getsign
        if np.isinf(d2logp_f1).sum()>0:
            print('inf Sec derivative f1')
            whichinf = np.isinf(d2logp_f1)
            getsign = np.sign(d2logp_f1[whichinf])
            d2logp_f1[whichinf] = lim_max
            d2logp_f1[whichinf] = d2logp_f1[whichinf] * getsign

        return d2logp_f0, d2logp_f1

    def mean(self, F, Y_metadata=None):
        ef0 = safe_exp(F[:, 0,None])
        phi = ef0 / (1 + ef0)
        phi = np.clip(phi, 1.0e-15, 1.0)  # numerical stability
        lamb_poisson = safe_exp(F[:, 1,None])
        lamb_poisson = np.clip(lamb_poisson, 0.0, 1.797e308)  # numerical stability
        #a = np.clip(a, 1e-9, 1e9)  # numerical stability
        #b = np.clip(b, 1e-9, 1e9)  # numerical stability
        mean = (1-phi)*lamb_poisson
        return mean

    def mean_sq(self,F, Y_metadata=None):
        ef0 = safe_exp(F[:, 0, None])
        phi = ef0 / (1 + ef0)
        phi = np.clip(phi, 1.0e-15, 1.0)  # numerical stability
        lamb_poisson = safe_exp(F[:, 1, None])
        lamb_poisson = np.clip(lamb_poisson, 0.0, 1.797e308)  # numerical stability
        mean = (1 - phi) * lamb_poisson
        mean_sq = np.square(mean)
        return mean_sq

    def variance(self,F, Y_metadata=None):
        ef0 = safe_exp(F[:, 0, None])
        phi = ef0 / (1 + ef0)
        phi = np.clip(phi, 1.0e-15, 1.0)  # numerical stability
        lamb_poisson = safe_exp(F[:, 1, None])
        lamb_poisson = np.clip(lamb_poisson, 0.0, 1.797e308)  # numerical stability
        # a = np.clip(a, 1e-9, 1e9)  # numerical stability
        # b = np.clip(b, 1e-9, 1e9)  # numerical stability
        var = (1 - phi) * lamb_poisson*(1+lamb_poisson*phi)
        return var

    def var_exp(self, y, M, V, gh_points=None, Y_metadata=None):
        # Variational Expectation
        # gh: Gaussian-Hermite quadrature
        if gh_points is None:
            gh_f, gh_w = self._gh_points(T=20)
        else:
            gh_f, gh_w = gh_points
        gh_w = gh_w / np.sqrt(np.pi)
        D = M.shape[1]
        expanded_F_tuples = []
        grid_tuple = [M.shape[0]]
        for d in range(D):
            grid_tuple.append(gh_f.shape[0])
            expanded_fd_tuple = [1] * (D + 1)
            expanded_fd_tuple[d + 1] = gh_f.shape[0]
            expanded_F_tuples.append(tuple(expanded_fd_tuple))

        # mean-variance tuple
        mv_tuple = [1] * (D + 1)
        mv_tuple[0] = M.shape[0]
        mv_tuple = tuple(mv_tuple)

        # building, normalizing and reshaping the grids
        F = np.zeros((reduce(lambda x, y: x * y, grid_tuple), D))
        for d in range(D):
            fd = np.zeros(tuple(grid_tuple))
            fd[:] = np.reshape(gh_f, expanded_F_tuples[d]) * np.sqrt(2 * np.reshape(V[:, d], mv_tuple)) \
                    + np.reshape(M[:, d], mv_tuple)
            F[:, d, None] = fd.reshape(reduce(lambda x, y: x * y, grid_tuple), -1, order='C')

        # function evaluation
        Y_full = np.repeat(y, gh_f.shape[0] ** D, axis=0)
        logp = self.logpdf(F, Y_full)
        logp = logp.reshape(tuple(grid_tuple))

        # calculating quadrature
        var_exp = logp.dot(gh_w)# / np.sqrt(np.pi)
        for d in range(D - 1):
            var_exp = var_exp.dot(gh_w)# / np.sqrt(np.pi)

        return var_exp[:, None]

    def var_exp_derivatives(self, y, M, V,GN=None, gh_points=None, Y_metadata=None):
        # Variational Expectation
        # gh: Gaussian-Hermite quadrature
        # GN: Use the Gauss-Newton Approximation
        if GN is None: GN = False

        if gh_points is None:
            gh_f, gh_w = self._gh_points(T=20)
        else:
            gh_f, gh_w = gh_points
        gh_w = gh_w / np.sqrt(np.pi)
        D = M.shape[1]
        expanded_F_tuples = []
        grid_tuple = [M.shape[0]]
        for d in range(D):
            grid_tuple.append(gh_f.shape[0])
            expanded_fd_tuple = [1] * (D + 1)
            expanded_fd_tuple[d + 1] = gh_f.shape[0]
            expanded_F_tuples.append(tuple(expanded_fd_tuple))

        # mean-variance tuple
        mv_tuple = [1] * (D + 1)
        mv_tuple[0] = M.shape[0]
        mv_tuple = tuple(mv_tuple)

        # building, normalizing and reshaping the grids
        F = np.zeros((reduce(lambda x, y: x * y, grid_tuple), D))
        for d in range(D):
            fd = np.zeros(tuple(grid_tuple))
            fd[:] = np.reshape(gh_f, expanded_F_tuples[d]) * np.sqrt(2 * np.reshape(V[:, d], mv_tuple)) \
                    + np.reshape(M[:, d], mv_tuple)
            F[:, d, None] = fd.reshape(reduce(lambda x, y: x * y, grid_tuple), -1, order='C')

        # function evaluation
        Y_full = np.repeat(y, gh_f.shape[0] ** D, axis=0)

        dlogp_a, dlogp_b = self.dlogp_df(F, Y_full)
        if np.isnan(dlogp_a).sum() > 0 or np.isnan(dlogp_b).sum() > 0:
            print('nan')

        GN = GN  # If True means we approximate the grads with Gauss-Newton so checkgrad(verbose) can change a bit
        if GN is False:
            d2logp_a, d2logp_b = self.d2logp_df2(F, Y_full)
            d2logp_a, d2logp_b = 0.5 * d2logp_a, 0.5 * d2logp_b
        else:
            d2logp_a, d2logp_b = -0.5 * dlogp_a.__pow__(2), -0.5 * dlogp_b.__pow__(2)

        dlogp_a = dlogp_a.reshape(tuple(grid_tuple))
        dlogp_b = dlogp_b.reshape(tuple(grid_tuple))
        d2logp_a = d2logp_a.reshape(tuple(grid_tuple))
        d2logp_b = d2logp_b.reshape(tuple(grid_tuple))

        ve_dm_fa = dlogp_a.dot(gh_w).dot(gh_w)# / np.square(np.sqrt(np.pi))
        ve_dm_fb = dlogp_b.dot(gh_w).dot(gh_w) #/ np.square(np.sqrt(np.pi))
        ve_dv_fa = d2logp_a.dot(gh_w).dot(gh_w) #/ np.square(np.sqrt(np.pi))
        ve_dv_fb = d2logp_b.dot(gh_w).dot(gh_w) #/ np.square(np.sqrt(np.pi))

        var_exp_dm = np.hstack((ve_dm_fa[:,None], ve_dm_fb[:,None]))
        var_exp_dv = np.hstack((ve_dv_fa[:,None], ve_dv_fb[:,None]))

        if np.isnan(var_exp_dm).sum() > 0 or np.isnan(var_exp_dv).sum() > 0:
            print('nan')

        return var_exp_dm, var_exp_dv

    def predictive(self, M, V, gh_points=None, Y_metadata=None):
        # Variational Expectation
        # gh: Gaussian-Hermite quadrature
        if gh_points is None:
            gh_f, gh_w = self._gh_points(T=20)
        else:
            gh_f, gh_w = gh_points
        gh_w = gh_w / np.sqrt(np.pi)
        D = M.shape[1]
        expanded_F_tuples = []
        grid_tuple = [M.shape[0]]
        for d in range(D):
            grid_tuple.append(gh_f.shape[0])
            expanded_fd_tuple = [1] * (D + 1)
            expanded_fd_tuple[d + 1] = gh_f.shape[0]
            expanded_F_tuples.append(tuple(expanded_fd_tuple))

        # mean-variance tuple
        mv_tuple = [1] * (D + 1)
        mv_tuple[0] = M.shape[0]
        mv_tuple = tuple(mv_tuple)

        # building, normalizing and reshaping the grids
        F = np.zeros((reduce(lambda x, y: x * y, grid_tuple), D))
        for d in range(D):
            fd = np.zeros(tuple(grid_tuple))
            fd[:] = np.reshape(gh_f, expanded_F_tuples[d]) * np.sqrt(2 * np.reshape(V[:, d], mv_tuple)) \
                    + np.reshape(M[:, d], mv_tuple)
            F[:, d, None] = fd.reshape(reduce(lambda x, y: x * y, grid_tuple), -1, order='C')

        mean = self.mean(F)
        mean = mean.reshape(tuple(grid_tuple))
        mean_pred = mean.dot(gh_w).dot(gh_w) #/ np.square(np.sqrt(np.pi))

        var = self.variance(F)
        var = var.reshape(tuple(grid_tuple))
        var_int = var.dot(gh_w).dot(gh_w) #/ np.square(np.sqrt(np.pi))
        mean_sq = self.mean_sq(F)
        mean_sq = mean_sq.reshape(tuple(grid_tuple))
        mean_sq_int = mean_sq.dot(gh_w).dot(gh_w) #/ np.square(np.sqrt(np.pi))

        var_pred = var_int + mean_sq_int - safe_square(mean_pred)
        return mean_pred[:,None] , var_pred[:,None]

    def log_predictive(self, Ytest, mu_F_star, v_F_star, num_samples):
        Ntest, D = mu_F_star.shape
        F_samples = np.empty((Ntest, D, num_samples))
        # function samples:
        for d in range(D):
            mu_fd_star = mu_F_star[:, d][:, None]
            var_fd_star = v_F_star[:, d][:, None]
            F_samples[:, d, :] = np.random.normal(mu_fd_star, np.sqrt(var_fd_star), size=(Ntest, num_samples))

        # monte-carlo:
        log_pred = -np.log(num_samples) + logsumexp(self.logpdf_sampling(F_samples, Ytest), axis=-1)
        log_pred = np.array(log_pred).reshape(*Ytest.shape)
        "I just changed this to have the log_predictive of each data point and not a mean values"
        #log_predictive = (1/num_samples)*log_pred.sum()

        return log_pred

    def get_metadata(self):
        dim_y = 1
        dim_f = 2
        dim_p = 1
        return dim_y, dim_f, dim_p

    def ismulti(self):
        # Returns if the distribution is multivariate
        return False
