# Code of Correlated Chained GPs with VIK was develped by Juan-Jose Giraldo
# based on the previous codes of the HetMOGP model by Pablo Moreno-Munoz (Universidad Carlos III)
# University of Sheffield

import numpy as np
import GPy
from convhetmogp_vik_fullcov.util import draw_mini_slices
from convhetmogp_vik_fullcov.conv_svmogp_inf import SVMOGPInf
from GPy.core.parameterization.param import Param
import matplotlib.pyplot as plt
from GPy.util import choleskies
from convhetmogp_vik_fullcov import util

from GPy.inference.latent_function_inference.posterior import Posterior
from collections import namedtuple
qu = namedtuple("q_U", "mu_u chols_u")
pu = namedtuple("p_U", "Kuu Luu Kuui")

class ConvHetMOGP_VIK(GPy.core.SparseGP):
    def __init__(self, X, Y, Z, kern_list_uq, kern_list_Gx,kern_list_Tq, likelihood, Y_metadata, name='ConvHetMOGP_VIK', batch_size=None):

        self.batch_size = batch_size
        self.kern_list = kern_list_uq
        self.likelihood = likelihood
        self.Y_metadata = Y_metadata
        self.kern_list_Gdj = kern_list_Gx
        self.kern_list_Tq = kern_list_Tq

        self.num_inducing = Z.shape[0]  # M
        self.num_latent_funcs = len(kern_list_uq) # Q
        self.num_output_funcs = likelihood.num_output_functions(self.Y_metadata)  #This is the number J in the paper
        self.W_list, self.kappa_list = util.random_W_kappas(self.num_latent_funcs, self.num_output_funcs, rank=1)

        check_ARD_uq = [kern.lengthscale.shape[0] > 1 for kern in kern_list_uq]
        check_ARD_Gx = [kern.lengthscale.shape[0] > 1 for kern in kern_list_Gx]  # This is just to verify Automatic Relevance Determination
        check_ARD_Tq = [kern.lengthscale.shape[0] > 1 for kern in kern_list_Tq]
        if (sum(check_ARD_uq) == 0) and (sum(check_ARD_Gx) == 0) and (sum(check_ARD_Tq) == 0):
            isARD = False
        elif (sum(check_ARD_uq) == check_ARD_uq.__len__()) and (sum(check_ARD_Gx) == check_ARD_Gx.__len__()) and (
                sum(check_ARD_Tq) == check_ARD_Tq.__len__()):
            isARD = True
        else:
            print('\nAll kernel_lists for Uq, Gx and Tx have to coincide in Automatic Relevance Determination,')
            print('All kernel_lists have to coincide: ARD=True or ARD=False\n')
            assert (sum(check_ARD_uq) == check_ARD_uq.__len__()) and (sum(check_ARD_Gx) == check_ARD_Gx.__len__()) and (sum(check_ARD_Tq) == check_ARD_Tq.__len__())

        self.kern_aux = GPy.kern.RBF(input_dim=Z.shape[1], lengthscale=1.0, variance=1.0, name='rbf_aux',ARD=isARD)+GPy.kern.White(input_dim=Z.shape[1])
        self.kern_aux.white.variance = 1e-6

        self.Xmulti = X
        self.Ymulti = Y

        # Batch the data
        self.Xmulti_all, self.Ymulti_all = X, Y
        if batch_size is None:
            #self.stochastic = False
            Xmulti_batch, Ymulti_batch = X, Y
        else:
            # Makes a climin slicer to make drawing minibatches much quicker
            #self.stochastic = False   #"This was True as Pablo had it"
            self.slicer_list = []
            [self.slicer_list.append(draw_mini_slices(Xmulti_task.shape[0], self.batch_size)) for Xmulti_task in self.Xmulti]
            Xmulti_batch, Ymulti_batch = self.new_batch()
            self.Xmulti, self.Ymulti = Xmulti_batch, Ymulti_batch

        # Initialize inducing points Z
        self.Xdim = Z.shape[1]
        Z = np.tile(Z,(1,self.num_latent_funcs))
        inference_method = SVMOGPInf()

        super(ConvHetMOGP_VIK, self).__init__(X=Xmulti_batch[0][1:10], Y=Ymulti_batch[0][1:10], Z=Z, kernel=kern_list_uq[0], likelihood=likelihood,
                                     mean_function=None, X_variance=None, inference_method=inference_method,
                                     Y_metadata=Y_metadata, name=name, normalizer=False)

        self.unlink_parameter(self.kern)  # Unlink SparseGP default param kernel


        _, self.B_list = util.LCM(input_dim=self.Xdim, output_dim=self.num_output_funcs, rank=1, kernels_list=self.kern_list,
                                  W_list=self.W_list, kappa_list=self.kappa_list)

        # Set-up optimization parameters: [Z, m_u, L_u]
        self.q_u_means = [Param('m_u'+str(dj), 10.0*np.random.randn(self.num_inducing, self.num_latent_funcs) +
                               10.0*np.tile(np.random.randn(1,self.num_latent_funcs),(self.num_inducing,1))) for dj in range(self.num_output_funcs)]
        chols = choleskies.triang_to_flat(np.tile(3*np.eye(self.num_inducing)[None,:,:], (self.num_latent_funcs,1,1)))
        self.q_u_chols = Param('L_u', chols)

        self.link_parameter(self.Z, index=0)
        [self.link_parameter(q_u_means) for q_u_means in self.q_u_means]
        self.link_parameters(self.q_u_chols)
        [self.link_parameter(kern_q) for kern_q in kern_list_uq]  # link all kernels
        [self.link_parameter(B_q) for B_q in self.B_list]
        [self.link_parameter(kern_list_Gjd) for kern_list_Gjd in kern_list_Gx]
        [self.link_parameter(kern_list_Tq) for kern_list_Tq in kern_list_Tq]
        #self.link_parameter(self.kern_aux.white.variance)

        self.vem_step = True # [True=VE-step, False=VM-step]
        self.ve_count = 0
        self.elbo = np.zeros((1,1))
        self.index_VEM = 0  #this is a variable to index correctly the self.elbo when using VEM
        self.Gauss_Newton = False  #This is a flag for using the Gauss-Newton approximation when dL_dV is needed

        for kern_q in self.kern_list:
            kern_q.variance = 1.0
            kern_q.variance.fix()
        for kern_Gjd in self.kern_list_Gdj:
            kern_Gjd.variance = 1.0
            kern_Gjd.variance.fix()
            #print('IN fix Gdj')
        for kern_Tq in self.kern_list_Tq:
            kern_Tq.variance = 1.0
            kern_Tq.variance.fix()

    def log_likelihood(self):
        return self._log_marginal_likelihood

    def test_log_likelihood(self,Xtest,Ytest):

        if not Xtest.__len__()==Ytest.__len__():
            print("The length of the input list has to coincide with the output list")
            return 0

        test_log_likelihood, _, _, _ = self.inference_method.inference(q_u_means=self.q_u_means,q_u_chols=self.q_u_chols, X=Xtest, Y=Ytest, Z=self.Z,kern_list=self.kern_list,kern_list_Gdj=self.kern_list_Gdj,kern_list_Tq=self.kern_list_Tq,kern_aux=self.kern_aux, likelihood=self.likelihood,B_list=self.B_list, Y_metadata=self.Y_metadata, batch_scale=None)
        return test_log_likelihood

    def parameters_changed(self):
        f_index = self.Y_metadata['function_index'].flatten()
        d_index = self.Y_metadata['d_index'].flatten()
        T = len(self.likelihood.likelihoods_list)
        self.batch_scale = []
        [self.batch_scale.append(float(self.Xmulti_all[t].shape[0])/float(self.Xmulti[t].shape[0])) for t in range(T)]
        self._log_marginal_likelihood, self.gradients, self.posteriors, _ = self.inference_method.inference(q_u_means=self.q_u_means,
                                                                     q_u_chols=self.q_u_chols, X=self.Xmulti, Y=self.Ymulti, Z=self.Z,
                                                                     kern_list=self.kern_list, kern_list_Gdj=self.kern_list_Gdj,kern_list_Tq=self.kern_list_Tq,kern_aux=self.kern_aux,likelihood=self.likelihood,
                                                                     B_list=self.B_list, Y_metadata=self.Y_metadata, batch_scale=self.batch_scale,Gauss_Newton=self.Gauss_Newton)
        J = self.likelihood.num_output_functions(self.Y_metadata) #here J is the total number latent parameter functions
        N = self.X.shape[0]
        M = self.num_inducing
        # # _, B_list = util.LCM(input_dim=self.Xdim, output_dim=D, rank=1, kernels_list=self.kern_list, W_list=self.W_list,
        # #                      kappa_list=self.kappa_list)
        Z_grad = np.zeros_like(self.Z.values)
        grad_kern_Gdj = [0.0]*J  #We create a list of length J for the gradients of smoothing kernels G(x)
        for q, kern_q in enumerate(self.kern_list):
            for d in range(J):
                self.q_u_means[d][:, q:q + 1].gradient = self.gradients['dL_dmu_u'][q][d]
            self.q_u_chols[:, q:q + 1].gradient = self.gradients['dL_dL_u'][q]

            # Update kernel hyperparameters: lengthscale and variance
            util.update_conv_Klambdalambda(kern_q, self.kern_list_Tq[q], self.kern_aux)
            self.kern_aux.update_gradients_full(self.gradients['dL_dKmm'][q],self.Z[:, q * self.Xdim:q * self.Xdim + self.Xdim])
            # grad_variance_kern_q = kern_q.variance.gradient.copy()
            if kern_q.use_invLengthscale is True:
                "To Do watch out"
                grad = kern_q.inv_lengthscale.gradient.copy()
            else:
                grad = self.kern_aux.rbf_aux.lengthscale.gradient.copy()
            if self.kern_list_Tq[q].use_invLengthscale is True:
                "To Do watch out"
                grad_Tq = kern_q.inv_lengthscale.gradient.copy()
            else:
                grad_Tq = 2.0 * self.kern_aux.rbf_aux.lengthscale.gradient.copy()

            # Update kernel hyperparameters: W + kappa
            Kffdiag = []
            KuqF = []
            for d in range(J):
                util.update_conv_Kff(kern_q, self.kern_list_Gdj[d], self.kern_aux)
                Kffdiag.append(self.kern_aux.Kdiag(self.Xmulti[f_index[d]]) * self.gradients['dL_dKdiag'][q][d] )
                #Kffdiag.append(self.gradients['dL_dKdiag'][q][d])   #old line
                #KuqF.append(self.gradients['dL_dKmn'][q][d] * kern_q.K(self.Z[:,q*self.Xdim:q*self.Xdim+self.Xdim], self.Xmulti[f_index[d]]))   #old line
                util.update_conv_Kflambda(kern_q, self.kern_list_Gdj[d], self.kern_list_Tq[q],self.kern_aux)
                KuqF.append(self.kern_aux.K(self.Z[:,q*self.Xdim:q*self.Xdim+self.Xdim], self.Xmulti[f_index[d]]) * self.gradients['dL_dKmn'][q][d])

            util.update_gradients_diag(self.B_list[q], Kffdiag)
            Bgrad = self.B_list[q].gradient.copy()
            util.update_gradients_Kmn(self.B_list[q], KuqF, J)
            Bgrad += self.B_list[q].gradient.copy()

            self.B_list[q].gradient = Bgrad.copy()

            for d in range(J):
                util.update_conv_Kflambda(kern_q, self.kern_list_Gdj[d],self.kern_list_Tq[q] ,self.kern_aux)
                self.kern_aux.update_gradients_full(self.B_list[q].W[d] * self.gradients['dL_dKmn'][q][d],self.Z[:, q * self.Xdim:q * self.Xdim + self.Xdim],self.Xmulti[f_index[d]])

                #dljq_dlambq = (self.kern_list_Gdj[d].lengthscale**2)/((self.kern_list_Gdj[d].lengthscale+ kern_q.lengthscale)**2)

                if kern_q.use_invLengthscale is True:
                    dE2jq_dlq = -0.5*(kern_q.inv_lengthscale**(-3.0/2.0))
                else:
                    dE2jq_dlq = 1.0
                grad += dE2jq_dlq * self.kern_aux.rbf_aux.lengthscale.gradient.copy()  # Juan wrote this

                if self.kern_list_Tq[q].use_invLengthscale is True:
                    "To Do watch out"
                    dE3_dIq = -0.5*(kern_q.inv_lengthscale**(-3.0/2.0))
                else:
                    dE3_dIq = 1.0
                grad_Tq += dE3_dIq * self.kern_aux.rbf_aux.lengthscale.gradient.copy()  # Juan wrote this

                #dljq_dPj = (kern_q.lengthscale ** 2)/((self.kern_list_Gdj[d].lengthscale + kern_q.lengthscale) ** 2)
                if self.kern_list_Gdj[d].use_invLengthscale is True:
                    dE2jq_dnj = -0.5*(self.kern_list_Gdj[d].inv_lengthscale**(-3.0/2.0))
                else:
                    dE2jq_dnj = 1.0
                grad_Gdj = dE2jq_dnj * self.kern_aux.rbf_aux.lengthscale.gradient.copy()

                util.update_conv_Kff(kern_q, self.kern_list_Gdj[d], self.kern_aux)
                self.kern_aux.update_gradients_diag(self.B_list[q].B[d,d] * self.gradients['dL_dKdiag'][q][d], self.Xmulti[f_index[d]])
                #dE2jjq_dlq = (self.kern_list_Gdj[d].lengthscale ** 2) / ((self.kern_list_Gdj[d].lengthscale + 2.0*kern_q.lengthscale) ** 2)

                if kern_q.use_invLengthscale is True:
                    dE1jjq_dlq = -0.5*(kern_q.inv_lengthscale**(-3.0/2.0))
                else:
                    dE1jjq_dlq = 1.0
                grad += dE1jjq_dlq * self.kern_aux.rbf_aux.lengthscale.gradient.copy()
                #dE1jjq_dnj = (2.0 * kern_q.lengthscale ** 2) / ((self.kern_list_Gdj[d].lengthscale + 2.0 * kern_q.lengthscale) ** 2)
                if self.kern_list_Gdj[d].use_invLengthscale is True:
                    dE1jjq_dnj = -1.0*(self.kern_list_Gdj[d].inv_lengthscale**(-3.0/2.0))
                else:
                    dE1jjq_dnj = 2.0
                grad_Gdj += dE1jjq_dnj * self.kern_aux.rbf_aux.lengthscale.gradient.copy()

                if q==0:
                    grad_kern_Gdj[d] = grad_Gdj.copy()
                else:
                    grad_kern_Gdj[d] += grad_Gdj.copy()


            #kern_q.gradient = grad  check if need copy() or not
            if kern_q.use_invLengthscale is True:
                kern_q.inv_lengthscale.gradient = grad.copy()
            else:
                kern_q.lengthscale.gradient = grad.copy()
            if self.kern_list_Tq[q].use_invLengthscale is True:
                self.kern_list_Tq[q].inv_lengthscale.gradient = grad_Tq.copy()
            else:
                self.kern_list_Tq[q].lengthscale.gradient = grad_Tq.copy()
            #kern_q.variance.gradient = grad_variance_kern_q.copy()

            if not self.Z.is_fixed:
                util.update_conv_Klambdalambda(kern_q, self.kern_list_Tq[q], self.kern_aux)
                Z_grad[:, q * self.Xdim:q * self.Xdim + self.Xdim] += self.kern_aux.gradients_X(self.gradients['dL_dKmm'][q], self.Z[:, q * self.Xdim:q * self.Xdim + self.Xdim]).copy()
                for d in range(J):
                    util.update_conv_Kflambda(kern_q, self.kern_list_Gdj[d], self.kern_list_Tq[q], self.kern_aux)
                    Z_grad[:, q * self.Xdim:q * self.Xdim + self.Xdim] += self.B_list[q].W[d] * self.kern_aux.gradients_X(self.gradients['dL_dKmn'][q][d], self.Z[:, q * self.Xdim:q * self.Xdim + self.Xdim],self.Xmulti[f_index[d]])
                    #Z_grad[:,q*self.Xdim:q*self.Xdim+self.Xdim] += kern_q.gradients_X(self.B_list[q].W[d]*self.gradients['dL_dKmn'][q][d], self.Z[:,q*self.Xdim:q*self.Xdim+self.Xdim], self.Xmulti[f_index[d]])

                #self.Z.gradient[:] = Z_grad
        self.Z.gradient[:] = Z_grad

        #We copy the contributions of gradients associated to each kernel_Gdj
        for d,kern_Gdj in enumerate(self.kern_list_Gdj):
            if kern_Gdj.use_invLengthscale is True:
                kern_Gdj.inv_lengthscale.gradient = grad_kern_Gdj[d].copy()
            else:
                kern_Gdj.lengthscale.gradient = grad_kern_Gdj[d].copy()

    def set_data(self, X, Y):
        """
        Set the data without calling parameters_changed to avoid wasted computation
        If this is called by the stochastic_grad function this will immediately update the gradients
        """
        self.Xmulti, self.Ymulti = X, Y

    def new_batch(self):
        """
        Return a new batch of X and Y by taking a chunk of data from the complete X and Y
        """
        T = len(self.likelihood.likelihoods_list)
        Xmulti_batch = []
        Ymulti_batch = []
        for t in range(T):
            i_task = next(self.slicer_list[t])
            Xmulti_batch.append(self.Xmulti_all[t][i_task])
            Ymulti_batch.append(self.Ymulti_all[t][i_task])
        return Xmulti_batch, Ymulti_batch

    def stochastic_grad(self, parameters):
        self.set_data(*self.new_batch())
        stochastic_gradients = self._grads(parameters)
        return stochastic_gradients

    def callback(self, i, max_iter, verbose=True, verbose_plot=False):
        ll = self.log_likelihood()
        self.elbo[self.index_VEM+i['n_iter'],0] =  self.log_likelihood()[0]
        if verbose:
            if i['n_iter']%50 ==0:
                print('svi - iteration '+str(i['n_iter'])+'/'+str(int(max_iter)))

        if verbose_plot:
            plt.ion()
            plt.show()
            plt.plot(i['n_iter'],ll,'k+')
            plt.draw()
            plt.pause(1e-5)

        if i['n_iter'] > max_iter:
            return True
        return False


    def posteriors_F(self, Xnew, which_out=None):
        # This function returns all the q(f*) associated to each output (It is the )
        # We assume that Xnew can be a list of length equal to the number of likelihoods defined for the HetMOGP
        # or Xnew can be a numpy array so that we can replicate it per each outout

        if isinstance(Xnew, list):
            Xmulti_all_new = Xnew
        else:
            Xmulti_all_new = []
            for i in range(self.num_output_funcs):
                Xmulti_all_new.append(Xnew.copy())

        M = self.Z.shape[0]
        Q = len(self.kern_list)
        D = self.likelihood.num_output_functions(self.Y_metadata)
        Kuu, Luu, Kuui = util.VIK_covariance(self.Z, self.kern_list,self.kern_list_Tq,self.kern_aux)
        p_U = pu(Kuu=Kuu, Luu=Luu, Kuui=Kuui)
        q_U = qu(mu_u=self.q_u_means, chols_u=self.q_u_chols)
        S_u = np.empty((Q, M, M))
        L_u = choleskies.flat_to_triang(q_U.chols_u.copy())
        [np.dot(L_u[q, :, :], L_u[q, :, :].T, S_u[q, :, :]) for q in range(Q)]

        # for every latent function f_d calculate q(f_d) and keep it as q(F):
        posteriors_F = []
        f_index = self.Y_metadata['function_index'].flatten()
        d_index = self.Y_metadata['d_index'].flatten()

        if which_out is None:
            indix_aux = f_index.copy()
        else:
            which_out = np.array(which_out)
            indix_aux = -1 * np.ones_like(f_index)
            for i in range(which_out.shape[0]):
                posix = np.where(f_index == which_out[i])
                indix_aux[posix] = f_index[posix].copy()

        for d in range(D):
            if f_index[d]==indix_aux[d]:
                Xtask = Xmulti_all_new[f_index[d]]
                q_fd, _ = self.inference_method.calculate_q_f(X=Xtask, Z=self.Z, q_U=q_U,S_u=S_u,p_U=p_U, kern_list=self.kern_list,kern_list_Gdj=self.kern_list_Gdj,kern_list_Tq=self.kern_list_Tq, kern_aux=self.kern_aux, B=self.B_list,
                                          M=M, N=Xtask.shape[0], Q=Q, D=D, d=d)
                # Posterior objects for output functions (used in prediction)
                posterior_fd = Posterior(mean=q_fd.m_fd.copy(), cov=q_fd.S_fd.copy(),
                                         K=util.function_covariance(X=Xtask, B=self.B_list, kernel_list=self.kern_list, d=d),
                                         prior_mean=np.zeros(q_fd.m_fd.shape))
                posteriors_F.append(posterior_fd)
            else:
                #posteriors_F.append(fake_posterior)
                posteriors_F.append([])
        return posteriors_F

    def predictive_f_star(self, Xnew, latent_function_ind=None, kern_list=None):
        #This function is just to select one of the predictive Posteriors_F_star associated to each output
        if latent_function_ind is None:
            latent_function_ind = 0

        posterior = self.posteriors_F(Xnew=Xnew)[latent_function_ind]
        return posterior.mean.copy(), np.diag(posterior.covariance).copy()[:,None]

    def predict(self, Xnew,which_out=None):
        """
        Description: Make a prediction of p(y*|y,Xnew)
        The function retuns two lists m_pred and v_pred with length equal to the number of outputs
        m_pred contains the prediction of each output
        v_pred contains the variance of each prediction
        """
        D = self.num_output_funcs #This D is the number of total latent functions that parameterize all Likelihooods; it is literally J= \sum Jd (in the paper)
        f_index = self.Y_metadata['function_index'].flatten()
        d_index = self.Y_metadata['d_index'].flatten()
        m_F_pred = []
        v_F_pred = []

        if isinstance(Xnew, list):
            Xpred = Xnew
        else:
            Xpred = []
            for i in range(self.num_output_funcs):
                Xpred.append(Xnew.copy())

        if which_out is None:
            indix_out = np.arange(len(self.likelihood.likelihoods_list))
        else:
            which_out = np.array(which_out)
            indix = np.arange(len(self.likelihood.likelihoods_list))
            indix_out = -1 * np.ones_like(indix)
            for i in range(which_out.shape[0]):
                posix = np.where(indix == which_out[i])
                indix_out[posix] = indix[posix]

        posteriors_F = self.posteriors_F(Xnew=Xpred,which_out=which_out)
        for t in range(len(self.likelihood.likelihoods_list)):
            if t == indix_out[t]:
                _,num_f_task,_ = self.likelihood.likelihoods_list[t].get_metadata()
                m_task_pred = np.empty((Xpred[t].shape[0], num_f_task))
                v_task_pred = np.empty((Xpred[t].shape[0], num_f_task))
                for d in range(D):
                    if f_index[d] == t:
                        #m_task_pred[:,d_index[d],None], v_task_pred[:,d_index[d],None] = self._raw_predict_f(Xpred[f_index[d]], output_function_ind=d)
                        m_task_pred[:, d_index[d], None], v_task_pred[:, d_index[d], None] = posteriors_F[d].mean.copy(), np.diag(posteriors_F[d].covariance.copy()) [:,None]

                m_F_pred.append(m_task_pred)
                v_F_pred.append(v_task_pred)
            else:
                m_F_pred.append([])
                v_F_pred.append([])

        m_pred, v_pred = self.likelihood.predictive(m_F_pred, v_F_pred, self.Y_metadata, indix_out=indix_out)
        return m_pred, v_pred

    def Log_Pred_Density(self, Xtest, Ytest, num_samples=1000):
        # The variables Xtest and Ytest have to be lists with the same length, and
        # each position Xtest[i] and Ytest[i] have to be N_i X D and N_i x 1 respectively,
        # where N_i is the number of data to test per output.
        # For instance N_1 (output 1) can be different to N_2 (output 2)

        f_index = self.Y_metadata['function_index'].flatten()
        T = len(self.Ymulti)
        mu_F_star = []
        v_F_star = []
        NLPD = np.zeros(T)

        posteriors_F = self.posteriors_F(Xnew=Xtest)
        for t in range(T):
            mu_F_star_task = np.empty((Ytest[t].shape[0],1))
            v_F_star_task = np.empty((Ytest[t].shape[0], 1))
            for d in range(self.num_output_funcs):
                if f_index[d] == t:
                    #m_fd_star, v_fd_star = self._raw_predict_f(Xtest[t], output_function_ind=d)
                    m_fd_star, v_fd_star = posteriors_F[d].mean.copy(), np.diag(posteriors_F[d].covariance.copy())[:, None]
                    mu_F_star_task = np.hstack((mu_F_star_task, m_fd_star))
                    v_F_star_task = np.hstack((v_F_star_task, v_fd_star))

            mu_F_star.append(mu_F_star_task[:,1:])
            v_F_star.append(v_F_star_task[:,1:])

        Log_Pred_Density = self.likelihood.log_predictive(Ytest, mu_F_star, v_F_star, Y_metadata=self.Y_metadata, num_samples=num_samples)

        return Log_Pred_Density

    def NLPD_loss(self, Xtest, Ytest, num_samples=1000):
        "Negative Log Predictive Density (NLPD)"
        "NLPD loss is the negative mean of all the log predictive: frac{-1}{N}\sum^N_{n=1} \log p(y_n=Ytest_n|Xtest_n)"
        NLPD_end = []
        NTasks = len(self.Ymulti)
        for task in range(NTasks):
            log_pred_accu = 0
            Ndata_to_test = Ytest[task].shape[0]
            if (Ndata_to_test>5000):
                batch_for_NLPD = 5000
            else:
                batch_for_NLPD = int(Ndata_to_test/2)
            how_many_batches = int(Ndata_to_test / batch_for_NLPD)  # 193 as a floor(969624/5000)
            start_final_batch = how_many_batches * batch_for_NLPD
            for i in range(how_many_batches):
                Xaux = []
                Yaux = []
                for taux in range(NTasks):
                    if taux == task:
                        Xaux.append(Xtest[task][i * batch_for_NLPD:i * batch_for_NLPD + batch_for_NLPD][:].copy())
                        Yaux.append(Ytest[task][i * batch_for_NLPD:i * batch_for_NLPD + batch_for_NLPD][:].copy())
                    else:
                        Xaux.append(Xtest[taux][0:1][:].copy())
                        Yaux.append(Ytest[taux][0:1][:].copy())
                log_pred_aux = self.Log_Pred_Density(Xaux,Yaux)
                log_pred_accu = log_pred_accu + log_pred_aux[task].sum()

            if (start_final_batch < Ndata_to_test):
                Xaux = []
                Yaux = []
                for taux in range(0, NTasks):
                    if taux == task:
                        Xaux.append(Xtest[task][start_final_batch:][:].copy())
                        Yaux.append(Ytest[task][start_final_batch:][:].copy())
                    else:
                        Xaux.append(Xtest[taux][0:1][:].copy())
                        Yaux.append(Ytest[taux][0:1][:].copy())
                log_pred_aux = self.Log_Pred_Density(Xaux,Yaux)
                log_pred_accu = log_pred_accu + log_pred_aux[task].sum()

            NLPD_end.append( -log_pred_accu / float(Ndata_to_test) )

        return NLPD_end