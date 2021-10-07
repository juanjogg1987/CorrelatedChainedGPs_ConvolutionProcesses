
import sys
import numpy as np
from GPy.likelihoods import link_functions
from GPy.likelihoods import Likelihood
from GPy.util.misc import safe_exp, safe_square
from scipy.special import beta, betaln, psi, zeta, gammaln, gamma, polygamma
from functools import reduce
from scipy.special import logsumexp


class NegBinomial(Likelihood):
    """
    Negative Binomial likelihood with latent functions over its parameters
    """

    def __init__(self, gp_link=None):
        if gp_link is None:
            gp_link = link_functions.Identity()

        super(NegBinomial, self).__init__(gp_link, name='NegBinomial')

    def pdf(self, F, y, Y_metadata=None):
        # exp(F[:,0]) is the mean
        # sigmoid(F[:,1]) is the parameter alpha=1/r
        mu = safe_exp(F[:, 0, None])
        alpha = safe_exp(F[:, 1, None]) / (safe_exp(F[:, 1, None]) + 1.0)
        r = 1.0 / alpha
        coef = gamma(y + r) / (gamma(y + 1) * gamma(r))
        pdf = coef * ((r / (r + mu)) ** r) * ((mu / (r + mu)) ** y)
        return pdf

    def logpdf(self, F, y, Y_metadata=None):
        mu = safe_exp(F[:, 0, None])
        alpha = safe_exp(F[:, 1, None]) / (safe_exp(F[:, 1, None]) + 1.0)
        r = 1.0 / alpha
        mu = np.clip(mu, 1e-9, 1e9)  # numerical stability
        r = np.clip(r, 1e-9, 1e9)  # numerical stability
        logpdf = gammaln(y+r) - gammaln(y+1) - gammaln(r) + r*(np.log(r) - np.log(r+mu)) + y*(np.log(mu) - np.log(r+mu))
        return logpdf

    def logpdf_sampling(self, F, y, Y_metadata=None):
        logpdf = np.zeros((y.shape[0], F.shape[2]))
        for i in range(F.shape[2]):
            logpdf[:, i] = self.logpdf(F[:, :, i], y, Y_metadata=None)[:, 0]

        return logpdf

    def samples(self, F ,num_samples, Y_metadata=None):
        mu = safe_exp(F[:, 0, None])
        alpha = safe_exp(F[:, 1, None]) / (safe_exp(F[:, 1, None]) + 1.0)
        r = 1.0 / alpha
        mu = np.clip(mu, 1e-9, 1e9)  # numerical stability
        r = np.clip(r, 1e-9, 1e9)  # numerical stability
        p = r / (mu + r)
        samples = np.random.negative_binomial(r, p)
        return samples

    def mean(self, F, Y_metadata=None):
        mu = safe_exp(F[:, 0, None])
        mu = np.clip(mu, 1e-9, 1e9)  # numerical stability
        mean = mu.copy()
        return mean

    def mean_sq(self, F, Y_metadata=None):
        mu = safe_exp(F[:, 0, None])
        mu = np.clip(mu, 1e-9, 1e9)  # numerical stability
        mean = mu.copy()
        mean_sq = np.square(mean)
        return mean_sq

    def variance(self, F, Y_metadata=None):
        mu = safe_exp(F[:, 0, None])
        alpha = safe_exp(F[:, 1, None]) / (safe_exp(F[:, 1, None]) + 1.0)
        r = 1.0 / alpha
        mu = np.clip(mu, 1e-9, 1e9)  # numerical stability
        r = np.clip(r, 1e-9, 1e9)  # numerical stability
        var = mu + (1.0 / r) * (mu ** 2)
        return var


    def dlogp_df(self, F, y, Y_metadata=None):
        mu = safe_exp(F[:, 0, None])
        alpha = safe_exp(F[:, 1, None]) / (safe_exp(F[:, 1, None]) + 1.0)
        r = 1.0 / alpha
        mu = np.clip(mu, 1e-9, 1e9)  # numerical stability
        r = np.clip(r, 1e-9, 1e9)  # numerical stability
        psi_yr = psi(y + r)
        psi_r = psi(r)
        dmu_df0 = mu.copy()
        dlogp_df0 = dmu_df0 * (-r / (r + mu) + y / mu - y / (r + mu))
        dalpha_df1 = alpha * (1 - alpha)
        dlogp_df1 = dalpha_df1*(-(r ** 2) * psi_yr + (r ** 2) * psi_r - (r ** 2) * np.log(r) + (r ** 2) * np.log(r + mu)+(r ** 3)/(
                            r + mu) + (r ** 2) * y / (r + mu) - r ** 2)
        return dlogp_df0, dlogp_df1


    def d2logp_df2(self, F, y, Y_metadata=None):
        mu = safe_exp(F[:, 0, None])
        alpha = safe_exp(F[:, 1, None]) / (safe_exp(F[:, 1, None]) + 1.0)
        r = 1.0 / alpha
        mu = np.clip(mu, 1e-9, 1e9)  # numerical stability
        r = np.clip(r, 1e-9, 1e9)  # numerical stability
        psi_yr = psi(y + r)
        psi_r = psi(r)
        dsig = alpha * (1 - alpha)
        C_prime = -(r) * (1 - alpha)
        A_prime = 2 * (r) * C_prime
        B_prime1 = -polygamma(1, y + r) * C_prime + polygamma(1, r) * C_prime + (dsig / alpha) + C_prime / (r + mu)
        B_prime2 = (C_prime * (r + mu) - (r + y) * (C_prime)) / ((r + mu) ** 2)
        B_prime = B_prime1 + B_prime2
        B = (-psi_yr + psi_r - np.log(r) + np.log(r + mu) + r / (r + mu) + y / (r + mu) - 1)
        d2logpdf_dalpha = A_prime * B + (r ** 2) * B_prime
        d2alpha_df1 = dsig * (1 - 2 * alpha)
        d2logp_df1 = d2alpha_df1 * (r ** 2) * B + dsig * d2logpdf_dalpha
        D = (-r / (r + mu) + y / mu - y / (r + mu))
        E = mu.copy()  # this is also E_prime sice mu = exp(f0) then E_prime=dmu_df0=exp(f0)
        D_prime = (r + y) * ((r + mu) ** (-2)) * E - y * (mu ** (-2)) * E
        d2logp_df0 = D_prime * E + D * E
        return d2logp_df0, d2logp_df1

    def var_exp(self, y, M, V, gh_points=None, Y_metadata=None):
        # Variational Expectation
        # gh: Gaussian-Hermite quadrature
        if gh_points is None:
            gh_f, gh_w = self._gh_points(T=16)
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

    def var_exp_derivatives(self, y, M, V, GN=None,gh_points=None, Y_metadata=None):
        # Variational Expectation
        # gh: Gaussian-Hermite quadrature
        # GN: Use the Gauss-Newton Approximation
        if GN is None: GN=False

        if gh_points is None:
            gh_f, gh_w = self._gh_points(T=16)
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
        GN = GN  # If True means we approximate the grads with Gauss-Newton so checkgrad(verbose) can change a bit
        if GN is False:
            d2logp_a, d2logp_b = self.d2logp_df2(F, Y_full)
            d2logp_a, d2logp_b = 0.5*d2logp_a, 0.5*d2logp_b
        else:
            d2logp_a, d2logp_b = -0.5 * dlogp_a.__pow__(2),-0.5 * dlogp_b.__pow__(2)

        dlogp_a = dlogp_a.reshape(tuple(grid_tuple))
        dlogp_b = dlogp_b.reshape(tuple(grid_tuple))
        d2logp_a = d2logp_a.reshape(tuple(grid_tuple))
        d2logp_b = d2logp_b.reshape(tuple(grid_tuple))

        ve_dm_fa = dlogp_a.dot(gh_w).dot(gh_w)# / np.square(np.sqrt(np.pi))
        ve_dm_fb = dlogp_b.dot(gh_w).dot(gh_w)# / np.square(np.sqrt(np.pi))
        ve_dv_fa = d2logp_a.dot(gh_w).dot(gh_w)# / np.square(np.sqrt(np.pi))
        ve_dv_fb = d2logp_b.dot(gh_w).dot(gh_w)# / np.square(np.sqrt(np.pi))

        var_exp_dm = np.hstack((ve_dm_fa[:,None], ve_dm_fb[:,None]))
        #var_exp_dv = 0.5*np.hstack((ve_dv_fa[:,None], ve_dv_fb[:,None]))
        var_exp_dv = np.hstack((ve_dv_fa[:, None], ve_dv_fb[:, None]))

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
        mean_pred = mean.dot(gh_w).dot(gh_w)# / np.square(np.sqrt(np.pi))

        var = self.variance(F)
        var = var.reshape(tuple(grid_tuple))
        var_int = var.dot(gh_w).dot(gh_w)# / np.square(np.sqrt(np.pi))
        mean_sq = self.mean_sq(F)
        mean_sq = mean_sq.reshape(tuple(grid_tuple))
        mean_sq_int = mean_sq.dot(gh_w).dot(gh_w)# / np.square(np.sqrt(np.pi))

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