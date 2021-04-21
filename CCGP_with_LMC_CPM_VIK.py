# The  Code of Correlated Chained GPs with Convoltion processes and Variational Inducing Kernels
# was develped by Juan-Jose Giraldo # University of Sheffield based on the previous codes of 
# the HetMOGP model by Pablo Moreno-Munoz (Universidad Carlos III)


import sys
import climin
import importlib
from functools import partial
import warnings
import os

sys.path.append('..')

import load_datasets_TransportPaper as pre
from importlib import reload

reload(pre)

import numpy as np
from scipy.stats import multinomial
from scipy.linalg.blas import dtrmm

import GPy
from GPy.util import choleskies
from GPy.core.parameterization.param import Param
from GPy.kern import Coregionalize
from GPy.likelihoods import Likelihood
from GPy.util import linalg

from likelihoods.poisson import Poisson
from likelihoods.bernoulli import Bernoulli
from likelihoods.gaussian import Gaussian
from likelihoods.categorical import Categorical
from likelihoods.hetgaussian import HetGaussian
from likelihoods.beta import Beta
from likelihoods.gamma import Gamma
from likelihoods.exponential import Exponential
from likelihoods.zipoisson import ZIPoisson

from hetmogp.util import draw_mini_slices
from hetmogp.het_likelihood import HetLikelihood
from hetmogp.svmogp import SVMOGP

from convhetmogp_vik_fullcov.CPVIK_hetmogp import ConvHetMOGP_VIK
from convhetmogp_fullcov.CP_hetmogp import ConvHetMOGP
from convhetmogp_vik_fullcov import util

reload(util)

from GPy.util.univariate_Gaussian import std_norm_cdf, std_norm_pdf
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
from matplotlib import rc, font_manager
from matplotlib import rcParams
import time
import getopt

warnings.filterwarnings("ignore")
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'
""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""

class commandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:], 'b:m:d:i:s:w:r:')
        # opts = dict(opts)
        # print(opts)
        self.minibatch = 100
        self.inducing = 20
        self.dimension = 1
        self.N_iter = 5000
        self.dataset = 'zip_conv1'
        self.which_model = 'CPM'
        self.which_seed = 107

        for op, arg in opts:
            # print(op,arg)
            if op == '-b':
                self.minibatch = arg
            if op == '-m':
                self.inducing = arg
            if op == '-d':
                self.dimension = arg
            if op == '-i':
                self.N_iter = arg
            if op == '-s':  # this is for (data)set
                self.dataset = arg
            if op == '-w':  # (w)hich model CGP, CCGP or HetMOGP or double_HetMOGP
                self.which_model = arg
            if op == '-r':  # (r)and seed
                self.which_seed = arg


""""""""""""""""""""""""""""""
config = commandLine()
#config.N_iter = 2000
num_inducing = int(config.inducing)  # number of inducing points
batch = int(config.minibatch)
input_dim = int(config.dimension)
""""""""""""""""""""""""""""""
if config.which_model=='CPM' or config.which_model=='VIK':
    convolved = True   #this is to run the Convolutional version
    print('\nModel to use: Convolution Processes Model\n')
else:
    convolved = False
    print('\nModel to use: Linear Model of Coregionalisation\n')
""""""""""""""""""""""""""""""
which_color = {
    "adam": '#d7191c',
    "vo": '#2b8cbe',
    "adad": '#fdbb84'
}

"""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""
methods = ['vo']
seeds = [int(config.which_seed)]
path_to_save = '/home/juanjo/Work_at_Home/My_codes_py/Codes_Github_Resubmit/Mock_local_experiments/'+config.which_model+'/'  #Work_at_Home
NLPD_adam = []
NLPD_fng = []
NLPD_adad = []
NLPD_hyb = []
ELBO_adam = []
ELBO_fng = []
ELBO_adad = []
ELBO_hyb = []

for myseed in seeds:
    plt.close('all')

    for posmeth, method in enumerate(methods):
        print('Running with Optimiser '+method)
        np.random.seed(101)
        import random

        random.seed(101)

        """"""""""""""""""""""""""""""""""""""
        def callback(i):
            global n_iter, start, Xtest, Ytest
            global ELBO, myTimes, NLPD

            ELBO.append(model.log_likelihood())
            myTimes.append(time.time())

            if (i['n_iter']) % 50 == 0:
                print(i['n_iter'])
                print(model.log_likelihood())
                #NLPD.append(model.negative_log_predictive(Xtest, Ytest, num_samples=1000))
            if i['n_iter'] > n_iter:
                myTimes = np.array(myTimes) - start
                return True
            return False


        """"""""""""""""""""""""""""""""""""""
        dataset = config.dataset
        Xtrain, Ytrain = pre.load_Hetdata(dataset=dataset, Ntoy=2000, Only_input_dim_toy=input_dim)
        """"""""""""""""""""""""""""""""""""""

        incomplete_out = np.inf
        # Likelihood Definition
        if dataset == 'zip_conv1':
            Q = 1
            mydict = {1: 10000, 2: 1000, 3: 500, 4: 100, 5: 50}
            q_s_ini = mydict.get(Xtrain[0].shape[1],10)
            prior_lamb = 1e-8
            likelihoods_list = [ZIPoisson()]
            #likelihoods_list = [Poisson()]
        elif dataset == 'zip_conv2':
            Q = 1
            mydict = {1: 10000, 2: 1000, 3: 500, 4: 100, 5: 50}
            q_s_ini = mydict.get(Xtrain[0].shape[1],10)
            prior_lamb = 1e-8
            likelihoods_list = [ZIPoisson()]
            #likelihoods_list = [Poisson()]
        elif dataset == 'toy1c':
            Q = 1
            prior_lamb = 1  #use 50 with Voptimisation
            my_proportion = [0.75, 0.75]
            #likelihoods_list = [Gaussian(sigma=0.1), Gaussian(sigma=0.1)]
            likelihoods_list = [HetGaussian(), HetGaussian()]
        elif dataset == 'toy2c':
            Q = 1
            prior_lamb = 1
            my_proportion = [0.75, 0.75, 0.75, 0.75, 0.75]
            likelihoods_list = [HetGaussian(), Beta(), Bernoulli(), Gamma(), Exponential()]
        elif dataset == 'toy3c':
            Q = 1
            prior_lamb = 1  #use 50 with Voptimisation
            my_proportion = [0.75, 0.75]
            #likelihoods_list = [Gaussian(sigma=0.1), Gaussian(sigma=0.1)]
            likelihoods_list = [HetGaussian(), HetGaussian()]

        if 'toy' in dataset:
            mydict = {1:10000,2:1000,3:500,4:100,5:50}
            q_s_ini = 0.5*mydict.get(Xtrain[0].shape[1],10)
            #q_s_ini = 10
            print('input D and q_s_ini',Xtrain[0].shape[1],q_s_ini)

        how_many_outs = 1

        np.random.seed(101)   #this same seed here is to guarrantee always the same data split for train and test
        _, Dim = Xtrain[0].shape
        Ntotal_with_test = []
        Ntotal_without_test = []
        index_train = []
        Ntotal_for_test = []
        for_train = 0.75
        for conti in range(likelihoods_list.__len__()):
            Ntotal_with_test.append(Xtrain[conti].shape[0])
            Ntotal_without_test.append(int(Ntotal_with_test[conti] * for_train))
            Ntotal_for_test.append( Ntotal_with_test[conti] - Ntotal_without_test[conti])
            index_train.append( np.random.permutation(np.arange(0, Ntotal_with_test[conti])) )

        rescale = 1
        Xtrain_new = [rescale * Xtrain[index_train[conti][0:Ntotal_without_test[conti]], :].copy() for conti,Xtrain in enumerate(Xtrain)] * how_many_outs
        Ytrain_new = [Ytrain[index_train[conti][0:Ntotal_without_test[conti]]].copy() for conti,Ytrain in enumerate(Ytrain)] * how_many_outs
        Xtest = [rescale * Xtrain[index_train[conti][Ntotal_without_test[conti]:], :].copy() for conti,Xtrain in enumerate(Xtrain)] * how_many_outs
        Ytest = [Ytrain[index_train[conti][Ntotal_without_test[conti]:]].copy() for conti,Ytrain in enumerate(Ytrain)] * how_many_outs

        Xtrain = Xtrain_new
        Ytrain = Ytrain_new

        likelihood = HetLikelihood(likelihoods_list)
        Y_metadata = likelihood.generate_metadata()

        # np.random.seed(101)
        myindex = []
        ind_split_aux = []

        # kmeans for selecting Z
        from scipy.cluster.vq import kmeans

        np.random.seed(myseed)
        # Z = 1.0 * kmeans(Xtrain, num_inducing)[0]

        minis = Xtrain[0].min(0)
        maxis = Xtrain[0].max(0)
        Dim = Xtrain[0].shape[1]
        Z = np.linspace(minis[0], maxis[0], num_inducing).reshape(1, -1)
        for i in range(Dim - 1):
            Zaux = np.linspace(minis[i + 1], maxis[i + 1], num_inducing)
            Z = np.concatenate((Z, Zaux[np.random.permutation(num_inducing)].reshape(1, -1)), axis=0)
            # Z = np.concatenate((Z, Zaux.reshape(1, -1)), axis=0)
        Z = 1.0 * Z.T

        n_iter = int(config.N_iter)
        all_NLPD = []
        Times_all = []
        ELBO_all = []

        random.seed(101)
        np.random.seed(101)
        ELBO = []
        NLPD = []
        myTimes = []

        Y = Ytrain.copy()
        X = Xtrain.copy()

        J = likelihood.num_output_functions(Y_metadata)  # This function indicates how many J latent functions we need
        print('J is:',J)
        np.random.seed(myseed)
        if not convolved:
            if ('toy' in dataset):
                ls_q = 1 * np.sqrt(Dim) * (np.random.rand(Q) + 0.001)  # 10 * np.ones(Q)   era 0.01
            elif ('zip_conv' in dataset):
                ls_q = 0.1 * np.sqrt(Dim) * (np.random.rand(Q) + 0.001)
            else:
                ls_q = 0.1 * np.sqrt(Dim) * (np.random.rand(Q) + 0.001)  # 10 * np.ones(Q)   era 0.01
            print("Initial lengthscales uq:", ls_q)
        else:
            if ('toy1c' in dataset):
                ls_q = 0.0001 * np.sqrt(Dim) * (np.random.rand(Q) + 0.001)  # 10 * np.ones(Q)   era 0.01
                lenghtscale = 0.1 * np.sqrt(Dim) * (np.random.rand(J) * np.random.rand(J))
                length_Tq = 0.1 * np.sqrt(Dim) * (np.random.rand(Q) * np.random.rand(Q))
            elif ('toy3c' in dataset):
                ls_q = 0.1 * np.sqrt(Dim) * (np.random.rand(Q) + 0.001)  # 10 * np.ones(Q)   era 0.01
                lenghtscale = 0.1 * np.sqrt(Dim) * (np.random.rand(J) * np.random.rand(J))
                length_Tq = 0.1 * np.sqrt(Dim) * (np.random.rand(Q) * np.random.rand(Q))
            elif ('zip_conv' in dataset):
                ls_q = 0.01 * np.sqrt(Dim) * (np.random.rand(Q) + 0.001)
                lenghtscale = 0.01 * np.sqrt(Dim) * (np.random.rand(J) * np.random.rand(J))
                length_Tq = 0.01 * np.sqrt(Dim) * (np.random.rand(Q) * np.random.rand(Q))
            else:
                ls_q = 0.1 * np.sqrt(Dim) * (np.random.rand(Q) + 0.001)  #
                lenghtscale = 0.1 * np.sqrt(Dim) * (np.random.rand(J) * np.random.rand(J))
            print("Initial lengthscales uq:", ls_q)
            print("Initial lengthscales smoothing Gdq:", lenghtscale)
            print("Initial lengthscales smoothing Tq:", length_Tq)

        chose_ARD = True
        var_q = 1.0 * np.ones(Q)  # use 0.1 for toy 0.1 0.5
        kern_list = util.latent_functions_prior(Q, lenghtscale=ls_q, variance=var_q, input_dim=Dim, ARD=chose_ARD)

        # MODEL
        if convolved:

            var_Tq = 1.0 * np.ones(Q)  # use 0.1 for toy 0.1 0.5
            kern_list_Tq = util.latent_functions_prior(Q, lenghtscale=length_Tq, variance=var_Tq, input_dim=Dim, ARD=chose_ARD,name='Tq')

            # We create the smoothing kernel for each latent function that parametrise the heterogeneous likelihoods
            kern_list_Gx = []
            variance = np.ones(J)  #
            for j in range(J):
                kern_j = GPy.kern.RBF(input_dim=Dim, lengthscale=lenghtscale[j], variance=variance[j], ARD=chose_ARD,
                                      inv_l=False)
                kern_j.name = 'kern_G' + str(j)
                kern_list_Gx.append(kern_j)

            if(config.which_model=='CPM'):
                model = ConvHetMOGP(X=X, Y=Y, Z=Z.copy(), kern_list_uq=kern_list, kern_list_Gx=kern_list_Gx,likelihood=likelihood, Y_metadata=Y_metadata, batch_size=batch)
            elif(config.which_model=='VIK'):
               model = ConvHetMOGP_VIK(X=X, Y=Y, Z=Z.copy(), kern_list_uq=kern_list,kern_list_Gx=kern_list_Gx,kern_list_Tq=kern_list_Tq, likelihood=likelihood, Y_metadata=Y_metadata,batch_size=batch)
        else:
            model = SVMOGP(X=X, Y=Y, Z=Z.copy(), kern_list=kern_list, likelihood=likelihood, Y_metadata=Y_metadata,batch_size=batch)

        #model.Z.fix()
        #model.q_u_chols.fix()
        #model.m_u0.fix()
        #model.m_u1.fix()
        #model.kern_list[0].fix()
        model['.*.kappa'].fix()

        """"""""""""""""""""""""""""""""""""""""""""

        """"""""""""""""""""""""""""""""""""""""""""

        for q in range(Q):
            model['B_q' + str(q) + '.W'] = 0.1 * np.random.randn(model['B_q0.W'].__len__())[:, None]
            model.kern_list[q].variance.fix()
        """"""""""""""""""""""""""""""""""""""""""""""""""""""
        print(model['B'])
        print('Initial Log Likelihood:\n',model.log_likelihood())

        if method == 'adam':
            opt = climin.Adam(model.optimizer_array, model.stochastic_grad, step_rate=0.005, decay_mom1=1 - 0.9,decay_mom2=1 - 0.999)
            ELBO.append(model.log_likelihood())
            #NLPD.append(model.negative_log_predictive(Xtest, Ytest, num_samples=1000))
            start = time.time()
            myTimes.append(start)
            print('Running Adam...')
            info = opt.minimize_until(callback)

        elif method == 'adad':
            opt = climin.Adadelta(model.optimizer_array, model.stochastic_grad, step_rate=0.005, momentum=0.9)
            ELBO.append(model.log_likelihood())
            #NLPD.append(model.negative_log_predictive(Xtest, Ytest, num_samples=1000))
            start = time.time()
            myTimes.append(start)
            print('Running Adadelta...')
            info = opt.minimize_until(callback)
        elif method == 'vo':
            model.Gauss_Newton = False
            opt = climin.VarOpt(model.optimizer_array, model.stochastic_grad, step_rate=0.005, s_ini=q_s_ini,decay_mom1=1 - 0.9,decay_mom2=1 - 0.999,prior_lambda=prior_lamb)
            ELBO.append(model.log_likelihood())
            #NLPD.append(model.negative_log_predictive(Xtest, Ytest, num_samples=1000))
            start = time.time()
            myTimes.append(start)
            print('Running Variationa Opt...')
            info = opt.minimize_until(callback)

        Times_all.append(np.array(myTimes).flatten())
        ELBO_all.append(np.array(ELBO).flatten())

        color = which_color[method]

        linewidth = 1.5

        d=0
        if(model.Xmulti_all[0].shape[1]==1):
            plt.rc('text', usetex=True)
            # plt.plot(Xtest[0],Ytest[0],'.',color='red')
            Xtest2 = []
            for d in range(likelihoods_list.__len__()):
                xneg = Xtrain[d].min()
                xpos = Xtrain[d].max()
                Xtest2.append(np.linspace(xneg, xpos, 1000)[:, None])

            mpred_c, vpred_c = model.predict(Xtest2)
            fonti = 13

            for d in range(likelihoods_list.__len__()):
                plt.figure(d)
                plt.plot(Xtest[d], Ytest[d], '.', color='red')

                plt.plot(X[d], Y[d], 'x', color='black')
                plt.plot(Xtest2[d], mpred_c[d], '-', color='blue', linewidth=2)
                if ("Poisson" in likelihoods_list[d]._name) or ("Gamma" in likelihoods_list[d]._name) or ("Beta" in likelihoods_list[d]._name):
                    plt.plot(Xtest2[d], np.clip(mpred_c[d] - 2.0 * np.sqrt(vpred_c[d]),0,np.inf), '--', color='blue')
                else:
                    plt.plot(Xtest2[d], mpred_c[d] - 2.0 * np.sqrt(vpred_c[d]), '--', color='blue')
                plt.plot(Xtest2[d], mpred_c[d] + 2.0 * np.sqrt(vpred_c[d]), '--', color='blue')
                plt.title(r'\bf{Prediction of CCGP+'+str(config.which_model)+' using a '+str(likelihoods_list[d]._name)+' Likelihood}', fontsize=fonti, fontweight="bold")
                # plt.ylim([Y[d].min(),Y[d].max()])
                if ("Poisson" in likelihoods_list[d]._name) or ("Gamma" in likelihoods_list[d]._name) or ("Beta" in likelihoods_list[d]._name):
                    plt.ylim([-2, np.max(Y[d])+0.3*np.max(Y[d])])
                else:
                    plt.ylim([np.min(Y[d]) - 0.3 * np.min(Y[d]), np.max(Y[d]) + 0.3 * np.max(Y[d])])

                plt.xlim([Xtrain[d].min(), Xtrain[d].max()])
                plt.xlabel(r'\bf{Input}', fontsize=fonti, fontweight="bold")
                plt.ylabel(r'\bf{Output}', fontsize=fonti, fontweight="bold")
                plt.legend((r'\bf{Test}', r'\bf{Train}', r'\bf{Prediction}',r'\bf{2 std}'), handlelength=1.5,fontsize=12)

        plt.figure(d+1)
        plt.semilogy(-ELBO_all[0], color, linewidth=linewidth, label=method)
        plt.title(r'\bf{Convergence of the Negative Evidence Lower Bound}',
            fontweight="bold")
        plt.xlabel(r'\bf{Iteration}', fontweight="bold")
        plt.ylabel(r'\bf{Neg ELBO}', fontweight="bold")
        # if (ELBO_all[0].max() > 0):
        #     plt.yscale('symlog')
        # else:
        #     plt.yscale('log')
        plt.grid(True, which='both')
        plt.gca().legend(methods, loc='upper center', ncol=4, bbox_to_anchor=(0.5, -0.13))
        plt.tight_layout()

        each = 50   #this is a variable to remember how frequently the NLPD is computed in the optimisation process

if (Xtest[0].shape[0] < 20000):
    mpred, vpred = model.predict(Xtest)
    nMSE_error = []
    for i in range(likelihoods_list.__len__()):
        nMSE_error.append(((mpred[i] - Ytest[i]) ** 2).mean() / np.var(Ytest[i]))

    print("\nThe normilised Mean Square Error per output:", nMSE_error)
    print("The total Error:", sum(nMSE_error))


"""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""

posteriors_Fj = model.posteriors_F(Xnew=X)
plt.figure(20)
plt.plot(X[0],posteriors_Fj[0].mean,'r.')
plt.plot(X[0],posteriors_Fj[1].mean,'r.')

print('\nNegative Log Predictive density metric over a test set:')
print(model.negative_log_predictive(Xtest,Ytest)[0])

