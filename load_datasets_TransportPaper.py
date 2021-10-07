import numpy as np
import sys
import scipy.io as sio
import GPy
import pandas as pd

sys.path.append('..')

from mlxtend.data import loadlocal_mnist
from convhetmogp_vik_fullcov import util
import matplotlib.pyplot as plt

np.random.seed(101)
import random
random.seed(101)

from likelihoods.bernoulli import Bernoulli
from likelihoods.gaussian import Gaussian
from likelihoods.hetgaussian import HetGaussian
from likelihoods.beta import Beta
from likelihoods.gamma import Gamma
from likelihoods.exponential import Exponential
from likelihoods.poisson import Poisson
from likelihoods.zipoisson import ZIPoisson
from likelihoods.negbinomial import NegBinomial


from hetmogp.het_likelihood import HetLikelihood

def load_bern(N=1000,input_dim=1):
    if input_dim==2:
        Nsqrt = int(N**(1.0/input_dim))
    print('input_dim:',input_dim)
    #Q = 5  # number of latent functions

    # Heterogeneous Likelihood Definition
    likelihoods_list = [Bernoulli()]  # Real + Binary
    likelihood = HetLikelihood(likelihoods_list)
    Y_metadata = likelihood.generate_metadata()
    D = likelihoods_list.__len__()
    Q = 1
    """"""""""""""""""""""""""""""
    rescale = 10
    Dim = input_dim
    if input_dim ==2:
        xy = rescale*np.linspace(-1.0, 1.0, Nsqrt)
        xx = rescale*np.linspace(-1.0, 1.0, Nsqrt)
        XX, XY = np.meshgrid(xx, xy)
        XX = XX.reshape(Nsqrt ** 2, 1)
        XY = XY.reshape(Nsqrt ** 2, 1)
        Xtoy = np.hstack((XX, XY))
    else:
        minis = -rescale * np.ones(Dim)
        maxis = rescale * np.ones(Dim)
        Xtoy = np.linspace(minis[0], maxis[0], N).reshape(1, -1)
        for i in range(Dim - 1):
            Xaux = np.linspace(minis[i + 1], maxis[i + 1], N)
            Xtoy = np.concatenate((Xtoy, Xaux[np.random.permutation(N)].reshape(1, -1)), axis=0)
            # Z = np.concatenate((Z, Zaux.reshape(1, -1)), axis=0)
        Xtoy = 1.0 * Xtoy.T


    def latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=None):
        if lenghtscale is None:
            lenghtscale = rescale*np.array([0.1]) #This is the one used for previous experiments
            #lenghtscale = np.array([1.0, 0.1, 0.2])
        else:
            lenghtscale = lenghtscale

        if variance is None:
            #variance = 1.0*np.random.rand(Q)
            variance = 1 * np.ones(Q)
        else:
            variance = variance

        kern_list = []
        for q in range(Q):
            #print("length:",lenghtscale[q])
            #print("var:", variance[q])
            kern_q = GPy.kern.RBF(input_dim=input_dim, lengthscale=lenghtscale[q], variance=variance[q], name='rbf')
            kern_q.name = 'kern_q' + str(q)
            kern_list.append(kern_q)
        return kern_list

    kern_list = latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=Dim)

    # True U and F functions
    def experiment_true_u_functions(kern_list, X):
        Q = kern_list.__len__()
        u_latent = np.zeros((X.shape[0], Q))
        """"""""""""""""""""""""""""""
        np.random.seed(103)
        """"""""""""""""""""""""""""""
        for q in range(Q):
            u_latent[:, q] = np.random.multivariate_normal(np.zeros(X.shape[0]), kern_list[q].K(X))

        return u_latent

    def experiment_true_f_functions(true_u, X_list, J):

        Q = true_u.shape[1]
        W = W_lincombination(Q, J)
        #
        f_j = np.zeros((X_list.shape[0], J))
        for q in range(Q):
            f_j += (W[q]*true_u[:,q]).T

        return f_j

    # True Combinations
    def W_lincombination(Q, J):
        W_list = []
        for q in range(Q):
            if q==0:
                W_list.append(np.array([2.7])[:,None])
            elif q == 1:
                W_list.append(np.array([1.4])[:,None])
            else:
                W_list.append(np.array([0.1])[:,None])

        return W_list

    """"""""""""""""""""""""""""""""

    # True functions values for inputs X
    f_index = Y_metadata['function_index'].flatten()
    J = f_index.__len__()
    trueU = experiment_true_u_functions(kern_list, Xtoy)
    trueF = experiment_true_f_functions(trueU, Xtoy,J)

    d_index = Y_metadata['d_index'].flatten()
    F_true = []


    for t in range(D):
        _, num_f_task, _ = likelihoods_list[t].get_metadata()
        f = np.empty((Xtoy.shape[0], num_f_task))
        for j in range(J):
            if f_index[j] == t:
                f[:, d_index[j], None] = trueF[:,j][:,None]

        F_true.append(f)

    # Generating training data Y (sampling from heterogeneous likelihood)
    Ytrain = likelihood.samples(F=F_true, Y_metadata=Y_metadata)
    Ytrain = [Ytrain[0].copy()]
    #print('Ya actualiza',Ytrain)
    Xtrain = []
    for d in range(likelihoods_list.__len__()):
        Xtrain.append(Xtoy.copy())

    return Xtrain, Ytrain

def load_ZIP(N=1000,input_dim=1):
    if input_dim==2:
        Nsqrt = int(N**(1.0/input_dim))
    print('input_dim:',input_dim)
    #Q = 5  # number of latent functions

    # Heterogeneous Likelihood Definition
    likelihoods_list = [ZIPoisson()]  # Real + Binary
    likelihood = HetLikelihood(likelihoods_list)
    Y_metadata = likelihood.generate_metadata()
    D = likelihoods_list.__len__()
    Q = 3
    """"""""""""""""""""""""""""""
    rescale = 1
    Dim = input_dim
    if input_dim ==2:
        xy = rescale*np.linspace(-1.0, 1.0, Nsqrt)
        xx = rescale*np.linspace(-1.0, 1.0, Nsqrt)
        XX, XY = np.meshgrid(xx, xy)
        XX = XX.reshape(Nsqrt ** 2, 1)
        XY = XY.reshape(Nsqrt ** 2, 1)
        Xtoy = np.hstack((XX, XY))
    else:
        minis = -rescale * np.ones(Dim)
        maxis = rescale * np.ones(Dim)
        Xtoy = np.linspace(minis[0], maxis[0], N).reshape(1, -1)
        for i in range(Dim - 1):
            Xaux = np.linspace(minis[i + 1], maxis[i + 1], N)
            Xtoy = np.concatenate((Xtoy, Xaux[np.random.permutation(N)].reshape(1, -1)), axis=0)
            # Z = np.concatenate((Z, Zaux.reshape(1, -1)), axis=0)
        Xtoy = 1.0 * Xtoy.T


    def latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=None):
        if lenghtscale is None:
            #lenghtscale = rescale*np.array([0.7,0.1]) #This is the one used for previous experiments
            lenghtscale = rescale*np.array([0.1, 0.7, 0.02])
        else:
            lenghtscale = lenghtscale

        if variance is None:
            #variance = 1.0*np.random.rand(Q)
            variance = 1 * np.ones(Q)
        else:
            variance = variance

        kern_list = []
        for q in range(Q):
            #print("length:",lenghtscale[q])
            #print("var:", variance[q])
            kern_q = GPy.kern.RBF(input_dim=input_dim, lengthscale=lenghtscale[q], variance=variance[q], name='rbf')
            kern_q.name = 'kern_q' + str(q)
            kern_list.append(kern_q)
        return kern_list

    kern_list = latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=Dim)

    # True U and F functions
    def experiment_true_u_functions(kern_list, X):
        Q = kern_list.__len__()
        u_latent = np.zeros((X.shape[0], Q))
        """"""""""""""""""""""""""""""
        np.random.seed(103)
        """"""""""""""""""""""""""""""
        for q in range(Q):
            u_latent[:, q] = np.random.multivariate_normal(np.zeros(X.shape[0]), kern_list[q].K(X))

        return u_latent

    def experiment_true_f_functions(true_u, X_list, J):

        Q = true_u.shape[1]
        W = W_lincombination(Q, J)
        #
        f_j = np.zeros((X_list.shape[0], J))
        for q in range(Q):
            f_j += (W[q]*true_u[:,q]).T

        return f_j

    # True Combinations
    def W_lincombination(Q, J):
        W_list = []
        for q in range(Q):
            if q==0:
                W_list.append(np.array([0.7,2.1])[:,None])
            elif q == 1:
                W_list.append(np.array([1.4,-0.4])[:,None])
            else:
                W_list.append(np.array([-0.1,0.9])[:,None])

        return W_list

    """"""""""""""""""""""""""""""""

    # True functions values for inputs X
    f_index = Y_metadata['function_index'].flatten()
    J = f_index.__len__()
    trueU = experiment_true_u_functions(kern_list, Xtoy)
    trueF = experiment_true_f_functions(trueU, Xtoy,J)

    d_index = Y_metadata['d_index'].flatten()
    F_true = []


    for t in range(D):
        _, num_f_task, _ = likelihoods_list[t].get_metadata()
        f = np.empty((Xtoy.shape[0], num_f_task))
        for j in range(J):
            if f_index[j] == t:
                f[:, d_index[j], None] = trueF[:,j][:,None]

        F_true.append(f)
    plt.figure(20)
    plt.plot(Xtoy[:,0],f[:,0],'.')
    plt.plot(Xtoy[:, 0], f[:, 1], '.')
    # Generating training data Y (sampling from heterogeneous likelihood)
    Ytrain = likelihood.samples(F=F_true, Y_metadata=Y_metadata)
    Ytrain = [Ytrain[0].copy()]
    #print('Ya actualiza',Ytrain)
    Xtrain = []
    for d in range(likelihoods_list.__len__()):
        Xtrain.append(Xtoy.copy())

    return Xtrain, Ytrain

def load_ZIP_conv1(N=1000, input_dim=1):
    if input_dim == 2:
        Nsqrt = int(N ** (1.0 / input_dim))
    print('input_dim:', input_dim)
    # Q = 5  # number of latent functions

    # Heterogeneous Likelihood Definition
    # likelihoods_list = [Gaussian(sigma=1.0), Bernoulli()] # Real + Binary
    likelihoods_list = [ZIPoisson()]  # Real + Binary
    # likelihoods_list = [Gaussian(sigma=1.0)]
    likelihood = HetLikelihood(likelihoods_list)
    Y_metadata = likelihood.generate_metadata()
    D = likelihoods_list.__len__()
    J = likelihood.num_output_functions(Y_metadata)
    Q = 2
    """"""""""""""""""""""""""""""
    rescale = 1
    Dim = input_dim
    if input_dim == 2:
        xy = rescale * np.linspace(-1.0, 1.0, Nsqrt)
        xx = rescale * np.linspace(-1.0, 1.0, Nsqrt)
        XX, XY = np.meshgrid(xx, xy)
        XX = XX.reshape(Nsqrt ** 2, 1)
        XY = XY.reshape(Nsqrt ** 2, 1)
        Xtoy = np.hstack((XX, XY))
    else:
        minis = -rescale * np.ones(Dim)
        maxis = rescale * np.ones(Dim)
        Xtoy = np.linspace(minis[0], maxis[0], N).reshape(1, -1)
        for i in range(Dim - 1):
            Xaux = np.linspace(minis[i + 1], maxis[i + 1], N)
            Xtoy = np.concatenate((Xtoy, Xaux[np.random.permutation(N)].reshape(1, -1)), axis=0)
            # Z = np.concatenate((Z, Zaux.reshape(1, -1)), axis=0)
        Xtoy = 1.0 * Xtoy.T

    def latent_functions_prior(Q, lengthscale=None, input_dim=None,name='kern_q'):
        #lenghtscale = rescale * np.array([0.5])  # This is the one used for previous experiments
        #lenghtscale = rescale * lengthscale
        lenghtscale = lengthscale
        variance = 1 * np.ones(Q)
        kern_list = []
        for q in range(Q):
            # print("length:",lenghtscale[q])
            # print("var:", variance[q])
            kern_q = GPy.kern.RBF(input_dim=input_dim, lengthscale=lenghtscale[q], variance=variance[q], name='rbf')
            kern_q.name = name + str(q)
            kern_list.append(kern_q)
        return kern_list


    #kern_list_uq = latent_functions_prior(Q,lengthscale=np.array([0.01]) ,input_dim=Dim,name='kern_q')
    kern_list_uq = latent_functions_prior(Q, lengthscale=np.array([0.01,0.05]), input_dim=Dim, name='kern_q')#old
    #kern_list_Gdj = latent_functions_prior(J, lengthscale=np.array([0.2, 0.02]), input_dim=Dim, name='kern_G')  #old
    kern_list_Gdj = latent_functions_prior(J, lengthscale=np.array([0.01, 0.001]), input_dim=Dim, name='kern_G')  # old
    #kern_list_uq = latent_functions_prior(Q, lengthscale=np.array([0.01,0.05,0.07]), input_dim=Dim, name='kern_q')
    #kern_list_Gdj = latent_functions_prior(J, lengthscale=np.array([0.02, 0.04]), input_dim=Dim, name='kern_G')
    kern_aux = GPy.kern.RBF(input_dim=Dim, lengthscale=1.0, variance=1.0, name='rbf_aux',ARD=False) + GPy.kern.White(input_dim=Dim)
    kern_aux.white.variance = 1e-6

    # True U and F functions
    def experiment_true_u_functions(kern_list,kern_list_Gdj, X):
        Q = kern_list.__len__()
        # for d,X in enumerate(X_list):
        u_latent = np.zeros((X.shape[0], Q))
        #np.random.seed(103)
        for q in range(Q):
            u_latent[:, q] = np.random.multivariate_normal(np.zeros(X.shape[0]), kern_list[q].K(X))

        return u_latent

    def experiment_true_f_functions(kern_u, kern_G, kern_aux,X,Q,J):
        np.random.seed(106)  #old
        #np.random.seed(105)
        W = W_lincombination(Q, J)
        # print(W)
        # for j in range(J):
        f_j = np.zeros((X.shape[0], J))
        #Cov_ff = np.zeros((X.shape[0], X.shape[0]))
        for j in range(J):
            Cov_ff = np.zeros((X.shape[0], X.shape[0]))
            for q in range(Q):
                util.update_conv_Kff(kern_u[q], kern_G[j], kern_aux)
                #f_j += (W[q] * true_u[:, q]).T
                #f_j[:,j] += W[q][j] * np.random.multivariate_normal(np.zeros(X.shape[0]), kern_aux.K(X))
                Cov_ff += (W[q][j]**2) *  kern_aux.K(X)
            f_j[:, j] = np.random.multivariate_normal(np.zeros(X.shape[0]), Cov_ff)
        # true_f.append(f_d)

        return f_j

    # True Combinations
    def W_lincombination(Q, J):
        W_list = []
        # q=1
        for q in range(Q):
            # W_list.append(np.array(([[-0.5], [0.1]])))
            if q == 0:
                # W_list.append(0.3*np.random.randn(J, 1))
                W_list.append(np.array([-0.5, 1.1])[:, None])
            elif q == 1:
                # W_list.append(2.0 * np.random.randn(J, 1))
                W_list.append(np.array([0.4,-0.3])[:, None])
            else:
                # W_list.append(10.0 * np.random.randn(J, 1)+0.1)
                W_list.append(np.array([0.1, -0.8])[:, None])
        return W_list

    """"""""""""""""""""""""""""""""

    # True functions values for inputs X
    f_index = Y_metadata['function_index'].flatten()
    J = f_index.__len__()
    trueF = experiment_true_f_functions(kern_list_uq, kern_list_Gdj, kern_aux,Xtoy,Q,J)
    # if input_dim==2:
    #     #from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    #
    #     from matplotlib import cm
    #     from matplotlib.ticker import LinearLocator, FormatStrFormatter
    #     fig = plt.figure()
    #     ax = fig.gca(projection='3d')
    #
    #     # Make data.
    #     # X = np.arange(-5, 5, 0.25)
    #     # Y = np.arange(-5, 5, 0.25)
    #     # X, Y = np.meshgrid(X, Y)
    #     # R = np.sqrt(X ** 2 + Y ** 2)
    #     # Z = np.sin(R)
    #
    #     # Plot the surface.
    #     surf = ax.plot_surface(Xtoy[:,0].reshape(Nsqrt,Nsqrt), Xtoy[:,1].reshape(Nsqrt,Nsqrt), trueF[:,2].reshape(Nsqrt,Nsqrt), cmap=cm.coolwarm,linewidth=0, antialiased=False)
    #
    #     # Customize the z axis.
    #     #ax.set_zlim(-1.01, 1.01)
    #     ax.zaxis.set_major_locator(LinearLocator(10))
    #     ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #
    #     # Add a color bar which maps values to colors.
    #     fig.colorbar(surf, shrink=0.5, aspect=5)
    #
    #     plt.show()
    #
    # else:
    #     plt.figure(15)
    #     plt.plot(trueF[:,1])
    #     plt.figure(16)
    #     plt.plot(trueU)

    d_index = Y_metadata['d_index'].flatten()
    F_true = []
    # for i,f_latent in enumerate(trueF):
    #    if

    for t in range(D):
        _, num_f_task, _ = likelihoods_list[t].get_metadata()
        f = np.empty((Xtoy.shape[0], num_f_task))
        for j in range(J):
            if f_index[j] == t:
                f[:, d_index[j], None] = trueF[:, j][:, None]

        F_true.append(f)

    plt.figure(20)
    plt.plot(Xtoy[:, 0], f[:, 0], '.')
    plt.plot(Xtoy[:, 0], f[:, 1], '.')

    # Generating training data Y (sampling from heterogeneous likelihood)
    Ytrain = likelihood.samples(F=F_true, Y_metadata=Y_metadata)
    Ytrain = [Ytrain[0].copy()]
    #Yreg0 = (Ytrain[0] - Ytrain[0].mean(0)) / (Ytrain[0].std(0))
    #Yreg1 = (Ytrain[1] - Ytrain[1].mean(0)) / (Ytrain[1].std(0))
    #Ytrain = [Yreg0, Yreg1]
    Xtrain = []
    for d in range(likelihoods_list.__len__()):
        Xtrain.append(Xtoy.copy())

    return Xtrain, Ytrain

def load_ZIP_conv2(N=1000, input_dim=1):
    if input_dim == 2:
        Nsqrt = int(N ** (1.0 / input_dim))
    print('input_dim:', input_dim)
    likelihoods_list = [ZIPoisson()]  # Real + Binary
    likelihood = HetLikelihood(likelihoods_list)
    Y_metadata = likelihood.generate_metadata()
    D = likelihoods_list.__len__()
    J = likelihood.num_output_functions(Y_metadata)
    Q = 3
    """"""""""""""""""""""""""""""
    rescale = 1
    Dim = input_dim
    if input_dim == 2:
        xy = rescale * np.linspace(-1.0, 1.0, Nsqrt)
        xx = rescale * np.linspace(-1.0, 1.0, Nsqrt)
        XX, XY = np.meshgrid(xx, xy)
        XX = XX.reshape(Nsqrt ** 2, 1)
        XY = XY.reshape(Nsqrt ** 2, 1)
        Xtoy = np.hstack((XX, XY))
    else:
        minis = -rescale * np.ones(Dim)
        maxis = rescale * np.ones(Dim)
        Xtoy = np.linspace(minis[0], maxis[0], N).reshape(1, -1)
        for i in range(Dim - 1):
            Xaux = np.linspace(minis[i + 1], maxis[i + 1], N)
            Xtoy = np.concatenate((Xtoy, Xaux[np.random.permutation(N)].reshape(1, -1)), axis=0)
            # Z = np.concatenate((Z, Zaux.reshape(1, -1)), axis=0)
        Xtoy = 1.0 * Xtoy.T

    def latent_functions_prior(Q, lengthscale=None, input_dim=None,name='kern_q'):
        lenghtscale = lengthscale
        variance = 1 * np.ones(Q)
        kern_list = []
        for q in range(Q):
            kern_q = GPy.kern.RBF(input_dim=input_dim, lengthscale=lenghtscale[q], variance=variance[q], name='rbf')
            kern_q.name = name + str(q)
            kern_list.append(kern_q)
        return kern_list

    kern_list_uq = latent_functions_prior(Q, lengthscale=np.array([0.001,0.005,0.007]), input_dim=Dim, name='kern_q')
    kern_list_Gdj = latent_functions_prior(J, lengthscale=np.array([0.3, 0.05]), input_dim=Dim, name='kern_G')
    kern_aux = GPy.kern.RBF(input_dim=Dim, lengthscale=1.0, variance=1.0, name='rbf_aux',ARD=False) + GPy.kern.White(input_dim=Dim)
    kern_aux.white.variance = 1e-6

    # True U and F functions
    def experiment_true_u_functions(kern_list,kern_list_Gdj, X):
        Q = kern_list.__len__()
        u_latent = np.zeros((X.shape[0], Q))
        for q in range(Q):
            u_latent[:, q] = np.random.multivariate_normal(np.zeros(X.shape[0]), kern_list[q].K(X))

        return u_latent

    def experiment_true_f_functions(kern_u, kern_G, kern_aux,X,Q,J):
        np.random.seed(105)  #old
        W = W_lincombination(Q, J)
        f_j = np.zeros((X.shape[0], J))
        for j in range(J):
            Cov_ff = np.zeros((X.shape[0], X.shape[0]))
            for q in range(Q):
                util.update_conv_Kff(kern_u[q], kern_G[j], kern_aux)
                Cov_ff += (W[q][j]**2) *  kern_aux.K(X)
            f_j[:, j] = np.random.multivariate_normal(np.zeros(X.shape[0]), Cov_ff)

        return f_j

    # True Combinations
    def W_lincombination(Q, J):
        W_list = []
        for q in range(Q):
            if q == 0:
                W_list.append(np.array([-0.5, 1.1])[:, None])
            elif q == 1:
                W_list.append(np.array([0.4,-0.3])[:, None])
            else:
                W_list.append(np.array([0.1, -0.8])[:, None])
        return W_list

    """"""""""""""""""""""""""""""""

    # True functions values for inputs X
    f_index = Y_metadata['function_index'].flatten()
    J = f_index.__len__()
    trueF = experiment_true_f_functions(kern_list_uq, kern_list_Gdj, kern_aux,Xtoy,Q,J)

    d_index = Y_metadata['d_index'].flatten()
    F_true = []

    for t in range(D):
        _, num_f_task, _ = likelihoods_list[t].get_metadata()
        f = np.empty((Xtoy.shape[0], num_f_task))
        for j in range(J):
            if f_index[j] == t:
                f[:, d_index[j], None] = trueF[:, j][:, None]

        F_true.append(f)

    # Generating training data Y (sampling from heterogeneous likelihood)
    Ytrain = likelihood.samples(F=F_true, Y_metadata=Y_metadata)
    Ytrain = [Ytrain[0].copy()]
    Xtrain = []
    for d in range(likelihoods_list.__len__()):
        Xtrain.append(Xtoy.copy())

    return Xtrain, Ytrain

def load_ZIP_conv3(N=1000, input_dim=1):
    if input_dim == 2:
        Nsqrt = int(N ** (1.0 / input_dim))
    print('input_dim:', input_dim)
    likelihoods_list = [ZIPoisson()]  # Real + Binary
    likelihood = HetLikelihood(likelihoods_list)
    Y_metadata = likelihood.generate_metadata()
    D = likelihoods_list.__len__()
    J = likelihood.num_output_functions(Y_metadata)
    Q = 1
    """"""""""""""""""""""""""""""
    rescale = 1
    Dim = input_dim
    if input_dim == 2:
        xy = rescale * np.linspace(-1.0, 1.0, Nsqrt)
        xx = rescale * np.linspace(-1.0, 1.0, Nsqrt)
        XX, XY = np.meshgrid(xx, xy)
        XX = XX.reshape(Nsqrt ** 2, 1)
        XY = XY.reshape(Nsqrt ** 2, 1)
        Xtoy = np.hstack((XX, XY))
    else:
        minis = -rescale * np.ones(Dim)
        maxis = rescale * np.ones(Dim)
        Xtoy = np.linspace(minis[0], maxis[0], N).reshape(1, -1)
        for i in range(Dim - 1):
            Xaux = np.linspace(minis[i + 1], maxis[i + 1], N)
            Xtoy = np.concatenate((Xtoy, Xaux[np.random.permutation(N)].reshape(1, -1)), axis=0)
            # Z = np.concatenate((Z, Zaux.reshape(1, -1)), axis=0)
        Xtoy = 1.0 * Xtoy.T

    def latent_functions_prior(Q, lengthscale=None, input_dim=None,name='kern_q'):
        lenghtscale = lengthscale
        variance = 1 * np.ones(Q)
        kern_list = []
        for q in range(Q):
            kern_q = GPy.kern.RBF(input_dim=input_dim, lengthscale=lenghtscale[q], variance=variance[q], name='rbf')
            kern_q.name = name + str(q)
            kern_list.append(kern_q)
        return kern_list


    kern_list_uq = latent_functions_prior(Q,lengthscale=np.array([0.01]) ,input_dim=Dim,name='kern_q')
    kern_list_Gdj = latent_functions_prior(J,lengthscale=np.array([0.2, 0.02]), input_dim=Dim, name='kern_G')
    kern_aux = GPy.kern.RBF(input_dim=Dim, lengthscale=1.0, variance=1.0, name='rbf_aux',ARD=False) + GPy.kern.White(input_dim=Dim)
    kern_aux.white.variance = 1e-6

    # True U and F functions
    def experiment_true_u_functions(kern_list,kern_list_Gdj, X):
        Q = kern_list.__len__()
        u_latent = np.zeros((X.shape[0], Q))
        np.random.seed(105)
        for q in range(Q):
            u_latent[:, q] = np.random.multivariate_normal(np.zeros(X.shape[0]), kern_list[q].K(X))

        return u_latent

    def experiment_true_f_functions(kern_u, kern_G, kern_aux,X,Q,J):

        W = W_lincombination(Q, J)
        f_j = np.zeros((X.shape[0], J))
        for j in range(J):
            for q in range(Q):
                util.update_conv_Kff(kern_u[q], kern_G[j], kern_aux)
                f_j[:,j] += W[q][j] * np.random.multivariate_normal(np.zeros(X.shape[0]), kern_aux.K(X))

        return f_j

    # True Combinations
    def W_lincombination(Q, J):
        W_list = []
        for q in range(Q):
            if q == 0:
                W_list.append(np.array([-0.5, 1.1])[:, None])
            elif q == 1:
                W_list.append(np.array([1.4, 0.3])[:, None])
            else:
                W_list.append(np.array([0.1, -0.8])[:, None])

        return W_list

    """"""""""""""""""""""""""""""""

    # True functions values for inputs X
    f_index = Y_metadata['function_index'].flatten()
    J = f_index.__len__()
    trueF = experiment_true_f_functions(kern_list_uq, kern_list_Gdj, kern_aux,Xtoy,Q,J)

    d_index = Y_metadata['d_index'].flatten()
    F_true = []

    for t in range(D):
        _, num_f_task, _ = likelihoods_list[t].get_metadata()
        f = np.empty((Xtoy.shape[0], num_f_task))
        for j in range(J):
            if f_index[j] == t:
                f[:, d_index[j], None] = trueF[:, j][:, None]

        F_true.append(f)

    # Generating training data Y (sampling from heterogeneous likelihood)
    Ytrain = likelihood.samples(F=F_true, Y_metadata=Y_metadata)
    Ytrain = [Ytrain[0].copy()]
    Xtrain = []
    for d in range(likelihoods_list.__len__()):
        Xtrain.append(Xtoy.copy())

    return Xtrain, Ytrain

def load_ZIP_conv4(N=1000, input_dim=1):
    if input_dim == 2:
        Nsqrt = int(N ** (1.0 / input_dim))
    print('input_dim:', input_dim)
    likelihoods_list = [ZIPoisson()]  # Real + Binary
    likelihood = HetLikelihood(likelihoods_list)
    Y_metadata = likelihood.generate_metadata()
    D = likelihoods_list.__len__()
    J = likelihood.num_output_functions(Y_metadata)
    Q = 1
    """"""""""""""""""""""""""""""
    rescale = 1
    Dim = input_dim
    if input_dim == 2:
        xy = rescale * np.linspace(-1.0, 1.0, Nsqrt)
        xx = rescale * np.linspace(-1.0, 1.0, Nsqrt)
        XX, XY = np.meshgrid(xx, xy)
        XX = XX.reshape(Nsqrt ** 2, 1)
        XY = XY.reshape(Nsqrt ** 2, 1)
        Xtoy = np.hstack((XX, XY))
    else:
        minis = -rescale * np.ones(Dim)
        maxis = rescale * np.ones(Dim)
        Xtoy = np.linspace(minis[0], maxis[0], N).reshape(1, -1)
        for i in range(Dim - 1):
            Xaux = np.linspace(minis[i + 1], maxis[i + 1], N)
            Xtoy = np.concatenate((Xtoy, Xaux[np.random.permutation(N)].reshape(1, -1)), axis=0)
        Xtoy = 1.0 * Xtoy.T

    def latent_functions_prior(Q, lengthscale=None, input_dim=None,name='kern_q'):
        lenghtscale = lengthscale
        variance = 1 * np.ones(Q)
        kern_list = []
        for q in range(Q):
            kern_q = GPy.kern.RBF(input_dim=input_dim, lengthscale=lenghtscale[q], variance=variance[q], name='rbf')
            kern_q.name = name + str(q)
            kern_list.append(kern_q)
        return kern_list


    kern_list_uq = latent_functions_prior(Q,lengthscale=np.array([0.01]) ,input_dim=Dim,name='kern_q')
    kern_list_Gdj = latent_functions_prior(J,lengthscale=np.array([0.01, 0.1]), input_dim=Dim, name='kern_G')
    kern_aux = GPy.kern.RBF(input_dim=Dim, lengthscale=1.0, variance=1.0, name='rbf_aux',ARD=False) + GPy.kern.White(input_dim=Dim)
    kern_aux.white.variance = 1e-6

    # True U and F functions
    def experiment_true_u_functions(kern_list,kern_list_Gdj, X):
        Q = kern_list.__len__()
        u_latent = np.zeros((X.shape[0], Q))
        np.random.seed(100)
        for q in range(Q):
            u_latent[:, q] = np.random.multivariate_normal(np.zeros(X.shape[0]), kern_list[q].K(X))

        return u_latent

    def experiment_true_f_functions(kern_u, kern_G, kern_aux,X,Q,J):

        W = W_lincombination(Q, J)
        f_j = np.zeros((X.shape[0], J))
        for j in range(J):
            for q in range(Q):
                util.update_conv_Kff(kern_u[q], kern_G[j], kern_aux)
                f_j[:,j] += W[q][j] * np.random.multivariate_normal(np.zeros(X.shape[0]), kern_aux.K(X))

        return f_j

    # True Combinations
    def W_lincombination(Q, J):
        W_list = []
        for q in range(Q):
            if q == 0:
                W_list.append(np.array([-1.5, 1.1])[:, None])
            elif q == 1:
                W_list.append(np.array([0.9, -0.3])[:, None])
            else:
                W_list.append(np.array([0.1, -0.8])[:, None])

        return W_list

    """"""""""""""""""""""""""""""""

    # True functions values for inputs X
    f_index = Y_metadata['function_index'].flatten()
    J = f_index.__len__()
    trueF = experiment_true_f_functions(kern_list_uq, kern_list_Gdj, kern_aux,Xtoy,Q,J)

    d_index = Y_metadata['d_index'].flatten()
    F_true = []

    for t in range(D):
        _, num_f_task, _ = likelihoods_list[t].get_metadata()
        f = np.empty((Xtoy.shape[0], num_f_task))
        for j in range(J):
            if f_index[j] == t:
                f[:, d_index[j], None] = trueF[:, j][:, None]

        F_true.append(f)

    # Generating training data Y (sampling from heterogeneous likelihood)
    Ytrain = likelihood.samples(F=F_true, Y_metadata=Y_metadata)
    Ytrain = [Ytrain[0].copy()]
    Xtrain = []
    for d in range(likelihoods_list.__len__()):
        Xtrain.append(Xtoy.copy())

    return Xtrain, Ytrain

def load_NB_conv1(N=1000, input_dim=1):
    if input_dim == 2:
        Nsqrt = int(N ** (1.0 / input_dim))
    print('input_dim:', input_dim)

    likelihoods_list = [NegBinomial()]  # Real + Binary
    likelihood = HetLikelihood(likelihoods_list)
    Y_metadata = likelihood.generate_metadata()
    D = likelihoods_list.__len__()
    J = likelihood.num_output_functions(Y_metadata)
    Q = 2
    """"""""""""""""""""""""""""""
    rescale = 1
    Dim = input_dim
    if input_dim == 2:
        xy = rescale * np.linspace(-1.0, 1.0, Nsqrt)
        xx = rescale * np.linspace(-1.0, 1.0, Nsqrt)
        XX, XY = np.meshgrid(xx, xy)
        XX = XX.reshape(Nsqrt ** 2, 1)
        XY = XY.reshape(Nsqrt ** 2, 1)
        Xtoy = np.hstack((XX, XY))
    else:
        minis = -rescale * np.ones(Dim)
        maxis = rescale * np.ones(Dim)
        Xtoy = np.linspace(minis[0], maxis[0], N).reshape(1, -1)
        for i in range(Dim - 1):
            Xaux = np.linspace(minis[i + 1], maxis[i + 1], N)
            Xtoy = np.concatenate((Xtoy, Xaux[np.random.permutation(N)].reshape(1, -1)), axis=0)
            # Z = np.concatenate((Z, Zaux.reshape(1, -1)), axis=0)
        Xtoy = 1.0 * Xtoy.T

    def latent_functions_prior(Q, lengthscale=None, input_dim=None,name='kern_q'):
        lenghtscale = lengthscale
        variance = 1 * np.ones(Q)
        kern_list = []
        for q in range(Q):
            kern_q = GPy.kern.RBF(input_dim=input_dim, lengthscale=lenghtscale[q], variance=variance[q], name='rbf')
            kern_q.name = name + str(q)
            kern_list.append(kern_q)
        return kern_list


    kern_list_uq = latent_functions_prior(Q, lengthscale=np.array([0.01,0.05]), input_dim=Dim, name='kern_q')
    kern_list_Gdj = latent_functions_prior(J, lengthscale=np.array([0.01, 0.001]), input_dim=Dim, name='kern_G')
    kern_aux = GPy.kern.RBF(input_dim=Dim, lengthscale=1.0, variance=1.0, name='rbf_aux',ARD=False) + GPy.kern.White(input_dim=Dim)
    kern_aux.white.variance = 1e-6

    # True U and F functions
    def experiment_true_u_functions(kern_list,kern_list_Gdj, X):
        Q = kern_list.__len__()
        u_latent = np.zeros((X.shape[0], Q))
        for q in range(Q):
            u_latent[:, q] = np.random.multivariate_normal(np.zeros(X.shape[0]), kern_list[q].K(X))

        return u_latent

    def experiment_true_f_functions(kern_u, kern_G, kern_aux,X,Q,J):
        np.random.seed(106)
        W = W_lincombination(Q, J)
        f_j = np.zeros((X.shape[0], J))
        for j in range(J):
            Cov_ff = np.zeros((X.shape[0], X.shape[0]))
            for q in range(Q):
                util.update_conv_Kff(kern_u[q], kern_G[j], kern_aux)
                Cov_ff += (W[q][j]**2) *  kern_aux.K(X)
            f_j[:, j] = np.random.multivariate_normal(np.zeros(X.shape[0]), Cov_ff)

        return f_j

    # True Combinations
    def W_lincombination(Q, J):
        W_list = []
        for q in range(Q):
            if q == 0:
                W_list.append(np.array([-0.5, 1.1])[:, None])
            elif q == 1:
                W_list.append(np.array([0.4,-0.3])[:, None])
            else:
                W_list.append(np.array([0.1, -0.8])[:, None])
        return W_list

    """"""""""""""""""""""""""""""""

    # True functions values for inputs X
    f_index = Y_metadata['function_index'].flatten()
    J = f_index.__len__()
    trueF = experiment_true_f_functions(kern_list_uq, kern_list_Gdj, kern_aux,Xtoy,Q,J)

    d_index = Y_metadata['d_index'].flatten()
    F_true = []

    for t in range(D):
        _, num_f_task, _ = likelihoods_list[t].get_metadata()
        f = np.empty((Xtoy.shape[0], num_f_task))
        for j in range(J):
            if f_index[j] == t:
                f[:, d_index[j], None] = trueF[:, j][:, None]

        F_true.append(f)

    # Generating training data Y (sampling from heterogeneous likelihood)
    Ytrain = likelihood.samples(F=F_true, Y_metadata=Y_metadata)
    Ytrain = [Ytrain[0].copy()]
    Xtrain = []
    for d in range(likelihoods_list.__len__()):
        Xtrain.append(Xtoy.copy())

    return Xtrain, Ytrain

def load_toy1(N=1000,input_dim=1):
    if input_dim==2:
        Nsqrt = int(N**(1.0/input_dim))
    print('input_dim:',input_dim)

    likelihoods_list = [HetGaussian(), Beta(),Bernoulli()]  # Real + Binary
    likelihood = HetLikelihood(likelihoods_list)
    Y_metadata = likelihood.generate_metadata()
    D = likelihoods_list.__len__()
    Q = 3
    """"""""""""""""""""""""""""""
    rescale = 10
    Dim = input_dim
    if input_dim ==2:
        xy = rescale*np.linspace(-1.0, 1.0, Nsqrt)
        xx = rescale*np.linspace(-1.0, 1.0, Nsqrt)
        XX, XY = np.meshgrid(xx, xy)
        XX = XX.reshape(Nsqrt ** 2, 1)
        XY = XY.reshape(Nsqrt ** 2, 1)
        Xtoy = np.hstack((XX, XY))
    else:
        minis = -rescale * np.ones(Dim)
        maxis = rescale * np.ones(Dim)
        Xtoy = np.linspace(minis[0], maxis[0], N).reshape(1, -1)
        for i in range(Dim - 1):
            Xaux = np.linspace(minis[i + 1], maxis[i + 1], N)
            Xtoy = np.concatenate((Xtoy, Xaux[np.random.permutation(N)].reshape(1, -1)), axis=0)
        Xtoy = 1.0 * Xtoy.T


    def latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=None):
        if lenghtscale is None:
            lenghtscale = rescale*np.array([0.5, 0.05, 0.1]) #This is the one used for previous experiments

        else:
            lenghtscale = lenghtscale

        if variance is None:
            variance = 1 * np.ones(Q)
        else:
            variance = variance

        kern_list = []
        for q in range(Q):
            kern_q = GPy.kern.RBF(input_dim=input_dim, lengthscale=lenghtscale[q], variance=variance[q], name='rbf')
            kern_q.name = 'kern_q' + str(q)
            kern_list.append(kern_q)
        return kern_list

    kern_list = latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=Dim)

    # True U and F functions
    def experiment_true_u_functions(kern_list, X):
        Q = kern_list.__len__()
        u_latent = np.zeros((X.shape[0], Q))
        np.random.seed(104)
        for q in range(Q):
            u_latent[:, q] = np.random.multivariate_normal(np.zeros(X.shape[0]), kern_list[q].K(X))

        return u_latent

    def experiment_true_f_functions(true_u, X_list, J):

        Q = true_u.shape[1]
        W = W_lincombination(Q, J)
        #
        f_j = np.zeros((X_list.shape[0], J))
        for q in range(Q):
            f_j += (W[q]*true_u[:,q]).T

        return f_j

    # True Combinations
    def W_lincombination(Q, J):
        W_list = []
        for q in range(Q):
            if q==0:
                W_list.append(np.array([-0.1,-0.1, 1.1, 2.1, -1.1])[:,None])
            elif q == 1:
                W_list.append(np.array([1.4, -0.5, 0.3, 0.7, 1.5])[:,None])
            else:
                W_list.append(np.array([0.1, -0.8, 1.3, 1.5, 0.5])[:,None])

        return W_list

    """"""""""""""""""""""""""""""""

    # True functions values for inputs X
    f_index = Y_metadata['function_index'].flatten()
    J = f_index.__len__()
    trueU = experiment_true_u_functions(kern_list, Xtoy)
    trueF = experiment_true_f_functions(trueU, Xtoy,J)

    d_index = Y_metadata['d_index'].flatten()
    F_true = []


    for t in range(D):
        _, num_f_task, _ = likelihoods_list[t].get_metadata()
        f = np.empty((Xtoy.shape[0], num_f_task))
        for j in range(J):
            if f_index[j] == t:
                f[:, d_index[j], None] = trueF[:,j][:,None]

        F_true.append(f)

    # Generating training data Y (sampling from heterogeneous likelihood)
    Ytrain = likelihood.samples(F=F_true, Y_metadata=Y_metadata)
    Yreg = (Ytrain[0] - Ytrain[0].mean(0)) / (Ytrain[0].std(0))
    Ytrain = [Yreg,np.clip(Ytrain[1],1.0e-9,0.99999),Ytrain[2]]
    Xtrain = []
    for d in range(likelihoods_list.__len__()):
        Xtrain.append(Xtoy)

    return Xtrain, Ytrain

def load_toy1_conv(N=1000, input_dim=1):
    if input_dim == 2:
        Nsqrt = int(N ** (1.0 / input_dim))
    print('input_dim:', input_dim)

    likelihoods_list = [Gaussian(sigma=0.1), Gaussian(sigma=0.1)]  # Real + Binary
    likelihood = HetLikelihood(likelihoods_list)
    Y_metadata = likelihood.generate_metadata()
    D = likelihoods_list.__len__()
    J = likelihood.num_output_functions(Y_metadata)
    Q = 1
    """"""""""""""""""""""""""""""
    rescale = 10
    Dim = input_dim
    if input_dim == 2:
        xy = rescale * np.linspace(-1.0, 1.0, Nsqrt)
        xx = rescale * np.linspace(-1.0, 1.0, Nsqrt)
        XX, XY = np.meshgrid(xx, xy)
        XX = XX.reshape(Nsqrt ** 2, 1)
        XY = XY.reshape(Nsqrt ** 2, 1)
        Xtoy = np.hstack((XX, XY))
    else:
        minis = -rescale * np.ones(Dim)
        maxis = rescale * np.ones(Dim)
        Xtoy = np.linspace(minis[0], maxis[0], N).reshape(1, -1)
        for i in range(Dim - 1):
            Xaux = np.linspace(minis[i + 1], maxis[i + 1], N)
            Xtoy = np.concatenate((Xtoy, Xaux[np.random.permutation(N)].reshape(1, -1)), axis=0)

        Xtoy = 1.0 * Xtoy.T

    def latent_functions_prior(Q, lengthscale=None, input_dim=None,name='kern_q'):
        lenghtscale = lengthscale
        variance = 1 * np.ones(Q)
        kern_list = []
        for q in range(Q):
            kern_q = GPy.kern.RBF(input_dim=input_dim, lengthscale=lenghtscale[q], variance=variance[q], name='rbf')
            kern_q.name = name + str(q)
            kern_list.append(kern_q)
        return kern_list


    kern_list_uq = latent_functions_prior(Q,lengthscale=np.array([0.1]) ,input_dim=Dim,name='kern_q')
    kern_list_Gdj = latent_functions_prior(J,lengthscale=np.array([0.3, 0.7]), input_dim=Dim, name='kern_G')
    kern_aux = GPy.kern.RBF(input_dim=Dim, lengthscale=1.0, variance=1.0, name='rbf_aux',ARD=False) + GPy.kern.White(input_dim=Dim)
    kern_aux.white.variance = 1e-6

    # True U and F functions
    def experiment_true_u_functions(kern_list,kern_list_Gdj, X):
        Q = kern_list.__len__()
        # for d,X in enumerate(X_list):
        u_latent = np.zeros((X.shape[0], Q))
        np.random.seed(104)
        for q in range(Q):
            u_latent[:, q] = np.random.multivariate_normal(np.zeros(X.shape[0]), kern_list[q].K(X))

        return u_latent

    def experiment_true_f_functions(kern_u, kern_G, kern_aux,X,Q,J):

        W = W_lincombination(Q, J)
        f_j = np.zeros((X.shape[0], J))
        for j in range(J):
            for q in range(Q):
                util.update_conv_Kff(kern_u[q], kern_G[j], kern_aux)
                f_j[:,j] += W[q][j] * np.random.multivariate_normal(np.zeros(X.shape[0]), kern_aux.K(X))

        return f_j

    # True Combinations
    def W_lincombination(Q, J):
        W_list = []
        for q in range(Q):
            if q == 0:
                W_list.append(np.array([-0.5, 2.1])[:, None])
            elif q == 1:
                W_list.append(np.array([1.4, 0.3,])[:, None])
            else:
                W_list.append(np.array([0.1, -0.8,])[:, None])

        return W_list

    """"""""""""""""""""""""""""""""

    # True functions values for inputs X
    f_index = Y_metadata['function_index'].flatten()
    J = f_index.__len__()
    trueF = experiment_true_f_functions(kern_list_uq, kern_list_Gdj, kern_aux,Xtoy,Q,J)
    d_index = Y_metadata['d_index'].flatten()
    F_true = []

    for t in range(D):
        _, num_f_task, _ = likelihoods_list[t].get_metadata()
        f = np.empty((Xtoy.shape[0], num_f_task))
        for j in range(J):
            if f_index[j] == t:
                f[:, d_index[j], None] = trueF[:, j][:, None]

        F_true.append(f)

    # Generating training data Y (sampling from heterogeneous likelihood)
    Ytrain = likelihood.samples(F=F_true, Y_metadata=Y_metadata)

    Xtrain = []
    for d in range(likelihoods_list.__len__()):
        Xtrain.append(Xtoy)

    return Xtrain, Ytrain

def load_toy2_conv(N=1000,input_dim=1):
    if input_dim==2:
        Nsqrt = int(N**(1.0/input_dim))
    print('input_dim:',input_dim)

    # Heterogeneous Likelihood Definition
    likelihoods_list = [HetGaussian(), Beta(), Bernoulli(), Gamma(), Exponential()]  # Real + Binary
    likelihood = HetLikelihood(likelihoods_list)
    Y_metadata = likelihood.generate_metadata()
    D = likelihoods_list.__len__()
    J = likelihood.num_output_functions(Y_metadata)
    Q = 1
    """"""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""
    rescale = 10
    Dim = input_dim
    if input_dim == 2:
        xy = rescale * np.linspace(-1.0, 1.0, Nsqrt)
        xx = rescale * np.linspace(-1.0, 1.0, Nsqrt)
        XX, XY = np.meshgrid(xx, xy)
        XX = XX.reshape(Nsqrt ** 2, 1)
        XY = XY.reshape(Nsqrt ** 2, 1)
        Xtoy = np.hstack((XX, XY))
    else:
        minis = -rescale * np.ones(Dim)
        maxis = rescale * np.ones(Dim)
        Xtoy = np.linspace(minis[0], maxis[0], N).reshape(1, -1)
        for i in range(Dim - 1):
            Xaux = np.linspace(minis[i + 1], maxis[i + 1], N)
            Xtoy = np.concatenate((Xtoy, Xaux[np.random.permutation(N)].reshape(1, -1)), axis=0)
            # Z = np.concatenate((Z, Zaux.reshape(1, -1)), axis=0)
        Xtoy = 1.0 * Xtoy.T

    def latent_functions_prior(Q, lengthscale=None, input_dim=None,name='kern_q'):
        #lenghtscale = rescale * np.array([0.5])  # This is the one used for previous experiments
        #lenghtscale = rescale * lengthscale
        lenghtscale = lengthscale
        variance = 1 * np.ones(Q)
        kern_list = []
        for q in range(Q):
            # print("length:",lenghtscale[q])
            # print("var:", variance[q])
            kern_q = GPy.kern.RBF(input_dim=input_dim, lengthscale=lenghtscale[q], variance=variance[q], name='rbf')
            kern_q.name = name + str(q)
            kern_list.append(kern_q)
        return kern_list

    kern_list_uq = latent_functions_prior(Q, lengthscale=np.array([0.1]), input_dim=Dim, name='kern_q')
    kern_list_Gdj = latent_functions_prior(J, lengthscale=np.array([0.3, 0.05,0.3, 0.6,0.05,1.4,0.3,0.9]), input_dim=Dim, name='kern_G')
    kern_aux = GPy.kern.RBF(input_dim=Dim, lengthscale=1.0, variance=1.0, name='rbf_aux', ARD=False) + GPy.kern.White(input_dim=Dim)
    kern_aux.white.variance = 1e-6

    # True U and F functions
    def experiment_true_u_functions(kern_list, X):
        Q = kern_list.__len__()
        u_latent = np.zeros((X.shape[0], Q))
        np.random.seed(104)
        for q in range(Q):
            u_latent[:, q] = np.random.multivariate_normal(np.zeros(X.shape[0]), kern_list[q].K(X))

        return u_latent

    def experiment_true_f_functions(kern_u, kern_G, kern_aux, X, Q, J):

        W = W_lincombination(Q, J)
        # print(W)
        # for j in range(J):
        f_j = np.zeros((X.shape[0], J))
        for j in range(J):
            for q in range(Q):
                util.update_conv_Kff(kern_u[q], kern_G[j], kern_aux)
                # f_j += (W[q] * true_u[:, q]).T
                f_j[:, j] += W[q][j] * np.random.multivariate_normal(np.zeros(X.shape[0]), kern_aux.K(X))
        # true_f.append(f_d)

        return f_j

    # True Combinations
    def W_lincombination(Q, J):
        W_list = []
        # q=1
        for q in range(Q):
            if q==0:
                W_list.append(np.array([-0.8,0.05, 0.6, 1.1, -1.1, -0.5, -0.6, -1.6])[:,None])
            elif q == 1:
                W_list.append(np.array([1.4, -0.5, 0.3, 0.7, 1.5, -0.3, 0.4, -0.2])[:,None])
            else:
                W_list.append(np.array([0.1, -0.8, 1.3, 1.5, 0.5,-0.02, 0.01, 0.5])[:,None])

        return W_list

    """"""""""""""""""""""""""""""""

    # True functions values for inputs X
    f_index = Y_metadata['function_index'].flatten()
    J = f_index.__len__()
    #trueU = experiment_true_u_functions(kern_list, Xtoy)
    trueF = experiment_true_f_functions(kern_list_uq, kern_list_Gdj, kern_aux,Xtoy,Q,J)

    d_index = Y_metadata['d_index'].flatten()
    F_true = []

    for t in range(D):
        _, num_f_task, _ = likelihoods_list[t].get_metadata()
        f = np.empty((Xtoy.shape[0], num_f_task))
        for j in range(J):
            if f_index[j] == t:
                f[:, d_index[j], None] = trueF[:,j][:,None]

        F_true.append(f)

    # Generating training data Y (sampling from heterogeneous likelihood)
    Ytrain = likelihood.samples(F=F_true, Y_metadata=Y_metadata)
    Yreg = (Ytrain[0] - Ytrain[0].mean(0)) / (Ytrain[0].std(0))
    Ytrain = [Yreg,np.clip(Ytrain[1],1.0e-9,0.99999),Ytrain[2],Ytrain[3],Ytrain[4]]
    Xtrain = []
    for d in range(likelihoods_list.__len__()):
        Xtrain.append(Xtoy)

    return Xtrain, Ytrain

def load_toy3_conv(N=1000, input_dim=1):
    if input_dim == 2:
        Nsqrt = int(N ** (1.0 / input_dim))
    print('input_dim:', input_dim)

    likelihoods_list = [HetGaussian(), HetGaussian()]  # Real + Binary
    likelihood = HetLikelihood(likelihoods_list)
    Y_metadata = likelihood.generate_metadata()
    D = likelihoods_list.__len__()
    J = likelihood.num_output_functions(Y_metadata)
    Q = 1
    """"""""""""""""""""""""""""""
    rescale = 5
    Dim = input_dim
    if input_dim == 2:
        xy = rescale * np.linspace(-1.0, 1.0, Nsqrt)
        xx = rescale * np.linspace(-1.0, 1.0, Nsqrt)
        XX, XY = np.meshgrid(xx, xy)
        XX = XX.reshape(Nsqrt ** 2, 1)
        XY = XY.reshape(Nsqrt ** 2, 1)
        Xtoy = np.hstack((XX, XY))
    else:
        minis = -rescale * np.ones(Dim)
        maxis = rescale * np.ones(Dim)
        Xtoy = np.linspace(minis[0], maxis[0], N).reshape(1, -1)
        for i in range(Dim - 1):
            Xaux = np.linspace(minis[i + 1], maxis[i + 1], N)
            Xtoy = np.concatenate((Xtoy, Xaux[np.random.permutation(N)].reshape(1, -1)), axis=0)
            # Z = np.concatenate((Z, Zaux.reshape(1, -1)), axis=0)
        Xtoy = 1.0 * Xtoy.T

    def latent_functions_prior(Q, lengthscale=None, input_dim=None,name='kern_q'):
        lenghtscale = lengthscale
        variance = 1 * np.ones(Q)
        kern_list = []
        for q in range(Q):
            kern_q = GPy.kern.RBF(input_dim=input_dim, lengthscale=lenghtscale[q], variance=variance[q], name='rbf')
            kern_q.name = name + str(q)
            kern_list.append(kern_q)
        return kern_list


    kern_list_uq = latent_functions_prior(Q,lengthscale=np.array([0.1]) ,input_dim=Dim,name='kern_q')
    kern_list_Gdj = latent_functions_prior(J,lengthscale=np.array([0.3, 0.7,0.05,0.1]), input_dim=Dim, name='kern_G')
    kern_aux = GPy.kern.RBF(input_dim=Dim, lengthscale=1.0, variance=1.0, name='rbf_aux',ARD=False) + GPy.kern.White(input_dim=Dim)
    kern_aux.white.variance = 1e-6

    # True U and F functions
    def experiment_true_u_functions(kern_list,kern_list_Gdj, X):
        Q = kern_list.__len__()
        u_latent = np.zeros((X.shape[0], Q))
        np.random.seed(104)
        for q in range(Q):
            u_latent[:, q] = np.random.multivariate_normal(np.zeros(X.shape[0]), kern_list[q].K(X))

        return u_latent

    def experiment_true_f_functions(kern_u, kern_G, kern_aux,X,Q,J):

        W = W_lincombination(Q, J)
        f_j = np.zeros((X.shape[0], J))
        for j in range(J):
            for q in range(Q):
                util.update_conv_Kff(kern_u[q], kern_G[j], kern_aux)
                f_j[:,j] += W[q][j] * np.random.multivariate_normal(np.zeros(X.shape[0]), kern_aux.K(X))

        return f_j

    # True Combinations
    def W_lincombination(Q, J):
        W_list = []
        for q in range(Q):
            if q == 0:
                W_list.append(np.array([-0.5, 1.1,0.9,-0.6])[:, None])
            elif q == 1:
                W_list.append(np.array([1.4, 0.3,0.8,1.2])[:, None])
            else:
                W_list.append(np.array([0.1, -0.8,0.5,-1.2])[:, None])

        return W_list

    """"""""""""""""""""""""""""""""

    # True functions values for inputs X
    f_index = Y_metadata['function_index'].flatten()
    J = f_index.__len__()
    trueF = experiment_true_f_functions(kern_list_uq, kern_list_Gdj, kern_aux,Xtoy,Q,J)
    d_index = Y_metadata['d_index'].flatten()
    F_true = []
    # for i,f_latent in enumerate(trueF):
    #    if

    for t in range(D):
        _, num_f_task, _ = likelihoods_list[t].get_metadata()
        f = np.empty((Xtoy.shape[0], num_f_task))
        for j in range(J):
            if f_index[j] == t:
                f[:, d_index[j], None] = trueF[:, j][:, None]

        F_true.append(f)

    # Generating training data Y (sampling from heterogeneous likelihood)
    Ytrain = likelihood.samples(F=F_true, Y_metadata=Y_metadata)

    Xtrain = []
    for d in range(likelihoods_list.__len__()):
        Xtrain.append(Xtoy)

    return Xtrain, Ytrain


def load_toy2(N=1000,input_dim=1):
    if input_dim==2:
        Nsqrt = int(N**(1.0/input_dim))
    print('input_dim:',input_dim)

    # Heterogeneous Likelihood Definition
    likelihoods_list = [HetGaussian(), Beta(), Bernoulli(), Gamma(), Exponential()]  # Real + Binary
    likelihood = HetLikelihood(likelihoods_list)
    Y_metadata = likelihood.generate_metadata()
    D = likelihoods_list.__len__()
    Q = 3
    """"""""""""""""""""""""""""""

    Dim = input_dim
    if input_dim ==2:
        xy = np.linspace(0.0, 1.0, Nsqrt)
        xx = np.linspace(0.0, 1.0, Nsqrt)
        XX, XY = np.meshgrid(xx, xy)
        XX = XX.reshape(Nsqrt ** 2, 1)
        XY = XY.reshape(Nsqrt ** 2, 1)
        Xtoy = np.hstack((XX, XY))
    else:
        minis = 0 * np.ones(Dim)
        maxis = 1 * np.ones(Dim)
        Xtoy = np.linspace(minis[0], maxis[0], N).reshape(1, -1)
        for i in range(Dim - 1):
            Xaux = np.linspace(minis[i + 1], maxis[i + 1], N)
            Xtoy = np.concatenate((Xtoy, Xaux[np.random.permutation(N)].reshape(1, -1)), axis=0)
            # Z = np.concatenate((Z, Zaux.reshape(1, -1)), axis=0)
        Xtoy = 1.0 * Xtoy.T


    def latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=None):
        if lenghtscale is None:
            lenghtscale = np.array([0.5, 0.05, 0.1]) #This is the one used for previous experiments
        else:
            lenghtscale = lenghtscale

        if variance is None:
            variance = 1 * np.ones(Q)
        else:
            variance = variance

        kern_list = []
        for q in range(Q):
            kern_q = GPy.kern.RBF(input_dim=input_dim, lengthscale=lenghtscale[q], variance=variance[q], name='rbf')
            kern_q.name = 'kern_q' + str(q)
            kern_list.append(kern_q)
        return kern_list

    kern_list = latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=Dim)

    # True U and F functions
    def experiment_true_u_functions(kern_list, X):
        Q = kern_list.__len__()
        u_latent = np.zeros((X.shape[0], Q))
        np.random.seed(104)
        for q in range(Q):
            u_latent[:, q] = np.random.multivariate_normal(np.zeros(X.shape[0]), kern_list[q].K(X))

        return u_latent

    def experiment_true_f_functions(true_u, X_list, J):
        #true_f = []

        Q = true_u.shape[1]
        W = W_lincombination(Q, J)
        f_j = np.zeros((X_list.shape[0], J))
        for q in range(Q):
            f_j += (W[q]*true_u[:,q]).T

        return f_j

    # True Combinations
    def W_lincombination(Q, J):
        W_list = []
        # q=1
        for q in range(Q):
            if q==0:
                W_list.append(np.array([-0.1,-0.1, 1.1, 2.1, -1.1, -0.5, -0.6, 0.1])[:,None])
            elif q == 1:
                W_list.append(np.array([1.4, -0.5, 0.3, 0.7, 1.5, -0.3, 0.4, -0.2])[:,None])
            else:
                W_list.append(np.array([0.1, -0.8, 1.3, 1.5, 0.5,-0.02, 0.01, 0.5])[:,None])

        return W_list

    """"""""""""""""""""""""""""""""

    # True functions values for inputs X
    f_index = Y_metadata['function_index'].flatten()
    J = f_index.__len__()
    trueU = experiment_true_u_functions(kern_list, Xtoy)
    trueF = experiment_true_f_functions(trueU, Xtoy,J)

    d_index = Y_metadata['d_index'].flatten()
    F_true = []

    for t in range(D):
        _, num_f_task, _ = likelihoods_list[t].get_metadata()
        f = np.empty((Xtoy.shape[0], num_f_task))
        for j in range(J):
            if f_index[j] == t:
                f[:, d_index[j], None] = trueF[:,j][:,None]

        F_true.append(f)

    # Generating training data Y (sampling from heterogeneous likelihood)
    Ytrain = likelihood.samples(F=F_true, Y_metadata=Y_metadata)
    Yreg = (Ytrain[0] - Ytrain[0].mean(0)) / (Ytrain[0].std(0))
    Ytrain = [Yreg,np.clip(Ytrain[1],1.0e-9,0.99999),Ytrain[2],Ytrain[3],Ytrain[4]]
    Xtrain = []
    for d in range(likelihoods_list.__len__()):
        Xtrain.append(Xtoy)

    return Xtrain, Ytrain

def load_toy3(N=1000,input_dim=1):
    if input_dim==2:
        Nsqrt = int(N**(1.0/input_dim))
    print('input_dim:',input_dim)

    # Heterogeneous Likelihood Definition
    likelihoods_list = [HetGaussian(), Beta(), Bernoulli(), Gamma(), Exponential(), Gaussian(sigma=0.1), Beta(), Bernoulli(), Gamma(), Exponential()]  # Real + Binary
    likelihood = HetLikelihood(likelihoods_list)
    Y_metadata = likelihood.generate_metadata()
    D = likelihoods_list.__len__()
    Q = 3
    """"""""""""""""""""""""""""""

    Dim = input_dim
    if input_dim ==2:
        xy = np.linspace(0.0, 1.0, Nsqrt)
        xx = np.linspace(0.0, 1.0, Nsqrt)
        XX, XY = np.meshgrid(xx, xy)
        XX = XX.reshape(Nsqrt ** 2, 1)
        XY = XY.reshape(Nsqrt ** 2, 1)
        Xtoy = np.hstack((XX, XY))
    else:
        minis = 0 * np.ones(Dim)
        maxis = 1 * np.ones(Dim)
        Xtoy = np.linspace(minis[0], maxis[0], N).reshape(1, -1)
        for i in range(Dim - 1):
            Xaux = np.linspace(minis[i + 1], maxis[i + 1], N)
            Xtoy = np.concatenate((Xtoy, Xaux[np.random.permutation(N)].reshape(1, -1)), axis=0)
            # Z = np.concatenate((Z, Zaux.reshape(1, -1)), axis=0)
        Xtoy = 1.0 * Xtoy.T


    def latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=None):
        if lenghtscale is None:
            lenghtscale = np.array([0.5, 0.05, 0.1]) #This is the one used for previous experiments
        else:
            lenghtscale = lenghtscale

        if variance is None:
            variance = 1 * np.ones(Q)
        else:
            variance = variance

        kern_list = []
        for q in range(Q):
            kern_q = GPy.kern.RBF(input_dim=input_dim, lengthscale=lenghtscale[q], variance=variance[q], name='rbf')
            kern_q.name = 'kern_q' + str(q)
            kern_list.append(kern_q)
        return kern_list

    kern_list = latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=Dim)

    # True U and F functions
    def experiment_true_u_functions(kern_list, X):
        Q = kern_list.__len__()
        u_latent = np.zeros((X.shape[0], Q))
        np.random.seed(104)
        for q in range(Q):
            u_latent[:, q] = np.random.multivariate_normal(np.zeros(X.shape[0]), kern_list[q].K(X))

        return u_latent

    def experiment_true_f_functions(true_u, X_list, J):

        Q = true_u.shape[1]
        W = W_lincombination(Q, J)
        #print(W)
        #for j in range(J):
        f_j = np.zeros((X_list.shape[0], J))
        for q in range(Q):
            f_j += (W[q]*true_u[:,q]).T
        #true_f.append(f_d)

        return f_j

    # True Combinations
    def W_lincombination(Q, J):
        W_list = []
        # q=1
        for q in range(Q):
            if q==0:
                W_list.append(np.array([-0.1,-0.1, 1.1, 2.1, -1.1, -0.5, -0.6, 0.1,  -1.1, 0.8, 1.5, -0.2, 0.05, 0.06, 0.3])[:,None])
            elif q == 1:
                W_list.append(np.array([1.4, -0.5, 0.3, 0.7, 1.5, -0.3, 0.4, -0.2,  0.4, 0.3, -0.7, -2.1, -0.03, 0.04, -0.5])[:,None])
            else:
                W_list.append(np.array([0.1, -0.8, 1.3, 1.5, 0.5,-0.02, 0.01, 0.5,  0.5, 1.0, 0.8, 3.0,0.1, -0.5, 0.4])[:,None])

        return W_list

    """"""""""""""""""""""""""""""""

    # True functions values for inputs X
    f_index = Y_metadata['function_index'].flatten()
    J = f_index.__len__()
    trueU = experiment_true_u_functions(kern_list, Xtoy)
    trueF = experiment_true_f_functions(trueU, Xtoy,J)

    d_index = Y_metadata['d_index'].flatten()
    F_true = []

    for t in range(D):
        _, num_f_task, _ = likelihoods_list[t].get_metadata()
        f = np.empty((Xtoy.shape[0], num_f_task))
        for j in range(J):
            if f_index[j] == t:
                f[:, d_index[j], None] = trueF[:,j][:,None]

        F_true.append(f)

    # Generating training data Y (sampling from heterogeneous likelihood)
    Ytrain = likelihood.samples(F=F_true, Y_metadata=Y_metadata)
    Yreg1 = (Ytrain[0] - Ytrain[0].mean(0)) / (Ytrain[0].std(0))
    Yreg2 = (Ytrain[5] - Ytrain[5].mean(0)) / (Ytrain[5].std(0))
    Ytrain = [Yreg1,np.clip(Ytrain[1],1.0e-9,0.99999),Ytrain[2],Ytrain[3],Ytrain[4],Yreg2,np.clip(Ytrain[6],1.0e-9,0.99999),Ytrain[7],Ytrain[8],Ytrain[9]]
    Xtrain = []
    for d in range(likelihoods_list.__len__()):
        Xtrain.append(Xtoy)

    return Xtrain, Ytrain

def load_toy4(N=1000,input_dim=1):
    if input_dim==2:
        Nsqrt = int(N**(1.0/input_dim))
    print('input_dim:',input_dim)

    likelihoods_list = [HetGaussian(), Beta()]  # Real + Binary
    likelihood = HetLikelihood(likelihoods_list)
    Y_metadata = likelihood.generate_metadata()
    D = likelihoods_list.__len__()
    Q = 3
    """"""""""""""""""""""""""""""

    Dim = input_dim
    if input_dim ==2:
        xy = np.linspace(0.0, 1.0, Nsqrt)
        xx = np.linspace(0.0, 1.0, Nsqrt)
        XX, XY = np.meshgrid(xx, xy)
        XX = XX.reshape(Nsqrt ** 2, 1)
        XY = XY.reshape(Nsqrt ** 2, 1)
        Xtoy = np.hstack((XX, XY))
    else:
        minis = 0 * np.ones(Dim)
        maxis = 1 * np.ones(Dim)
        Xtoy = np.linspace(minis[0], maxis[0], N).reshape(1, -1)
        for i in range(Dim - 1):
            Xaux = np.linspace(minis[i + 1], maxis[i + 1], N)
            Xtoy = np.concatenate((Xtoy, Xaux[np.random.permutation(N)].reshape(1, -1)), axis=0)
        Xtoy = 1.0 * Xtoy.T


    def latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=None):
        if lenghtscale is None:
            lenghtscale = np.array([0.5, 0.05, 0.1]) #This is the one used for previous experiments
        else:
            lenghtscale = lenghtscale

        if variance is None:
            variance = 1 * np.ones(Q)
        else:
            variance = variance

        kern_list = []
        for q in range(Q):
            kern_q = GPy.kern.RBF(input_dim=input_dim, lengthscale=lenghtscale[q], variance=variance[q], name='rbf')
            kern_q.name = 'kern_q' + str(q)
            kern_list.append(kern_q)
        return kern_list

    kern_list = latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=Dim)

    # True U and F functions
    def experiment_true_u_functions(kern_list, X):
        Q = kern_list.__len__()
        u_latent = np.zeros((X.shape[0], Q))
        np.random.seed(104)
        for q in range(Q):
            u_latent[:, q] = np.random.multivariate_normal(np.zeros(X.shape[0]), kern_list[q].K(X))

        return u_latent

    def experiment_true_f_functions(true_u, X_list, J):
        #true_f = []

        Q = true_u.shape[1]
        W = W_lincombination(Q, J)
        #print(W)
        #for j in range(J):
        f_j = np.zeros((X_list.shape[0], J))
        for q in range(Q):
            f_j += (W[q]*true_u[:,q]).T
        #true_f.append(f_d)

        return f_j

    # True Combinations
    def W_lincombination(Q, J):
        W_list = []
        # q=1
        for q in range(Q):
            if q==0:
                W_list.append(np.array([-0.1,-0.1,1.1,2.1])[:,None])
            elif q == 1:
                W_list.append(np.array([1.4, -0.5, 0.3, 0.7])[:,None])
            else:
                W_list.append(np.array([0.1, -0.8, 1.3, 1.5])[:,None])

        return W_list

    """"""""""""""""""""""""""""""""

    # True functions values for inputs X
    f_index = Y_metadata['function_index'].flatten()
    J = f_index.__len__()
    trueU = experiment_true_u_functions(kern_list, Xtoy)
    trueF = experiment_true_f_functions(trueU, Xtoy,J)

    d_index = Y_metadata['d_index'].flatten()
    F_true = []

    for t in range(D):
        _, num_f_task, _ = likelihoods_list[t].get_metadata()
        f = np.empty((Xtoy.shape[0], num_f_task))
        for j in range(J):
            if f_index[j] == t:
                f[:, d_index[j], None] = trueF[:,j][:,None]

        F_true.append(f)

    # Generating training data Y (sampling from heterogeneous likelihood)
    Ytrain = likelihood.samples(F=F_true, Y_metadata=Y_metadata)
    Yreg = (Ytrain[0] - Ytrain[0].mean(0)) / (Ytrain[0].std(0))
    Ytrain = [Yreg,np.clip(Ytrain[1],1.0e-9,0.99999)]
    Xtrain = []
    for d in range(likelihoods_list.__len__()):
        Xtrain.append(Xtoy)

    return Xtrain, Ytrain

def load_toy5(N=1000,input_dim=1):
    if input_dim==2:
        Nsqrt = int(N**(1.0/input_dim))
    print('input_dim:',input_dim)
    #Q = 5  # number of latent functions

    # Heterogeneous Likelihood Definition
    # likelihoods_list = [Gaussian(sigma=1.0), Bernoulli()] # Real + Binary
    likelihoods_list = [HetGaussian(), Beta(), Gamma()]  # Real + Binary
    # likelihoods_list = [Gaussian(sigma=1.0)]
    likelihood = HetLikelihood(likelihoods_list)
    Y_metadata = likelihood.generate_metadata()
    D = likelihoods_list.__len__()
    Q = 3
    """"""""""""""""""""""""""""""

    Dim = input_dim
    if input_dim ==2:
        xy = np.linspace(0.0, 1.0, Nsqrt)
        xx = np.linspace(0.0, 1.0, Nsqrt)
        XX, XY = np.meshgrid(xx, xy)
        XX = XX.reshape(Nsqrt ** 2, 1)
        XY = XY.reshape(Nsqrt ** 2, 1)
        Xtoy = np.hstack((XX, XY))
    else:
        minis = 0 * np.ones(Dim)
        maxis = 1 * np.ones(Dim)
        Xtoy = np.linspace(minis[0], maxis[0], N).reshape(1, -1)
        for i in range(Dim - 1):
            Xaux = np.linspace(minis[i + 1], maxis[i + 1], N)
            Xtoy = np.concatenate((Xtoy, Xaux[np.random.permutation(N)].reshape(1, -1)), axis=0)
            # Z = np.concatenate((Z, Zaux.reshape(1, -1)), axis=0)
        Xtoy = 1.0 * Xtoy.T


    def latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=None):
        if lenghtscale is None:
            lenghtscale = np.array([0.5, 0.05, 0.1]) #This is the one used for previous experiments
            #lenghtscale = np.array([1.0, 0.1, 0.2])
        else:
            lenghtscale = lenghtscale

        if variance is None:
            #variance = 1.0*np.random.rand(Q)
            variance = 1 * np.ones(Q)
        else:
            variance = variance

        kern_list = []
        for q in range(Q):
            #print("length:",lenghtscale[q])
            #print("var:", variance[q])
            kern_q = GPy.kern.RBF(input_dim=input_dim, lengthscale=lenghtscale[q], variance=variance[q], name='rbf')
            kern_q.name = 'kern_q' + str(q)
            kern_list.append(kern_q)
        return kern_list

    kern_list = latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=Dim)

    # True U and F functions
    def experiment_true_u_functions(kern_list, X):
        Q = kern_list.__len__()
        # for d,X in enumerate(X_list):
        u_latent = np.zeros((X.shape[0], Q))
        np.random.seed(104)
        for q in range(Q):
            u_latent[:, q] = np.random.multivariate_normal(np.zeros(X.shape[0]), kern_list[q].K(X))

        return u_latent

    def experiment_true_f_functions(true_u, X_list, J):
        #true_f = []

        Q = true_u.shape[1]
        W = W_lincombination(Q, J)
        #print(W)
        #for j in range(J):
        f_j = np.zeros((X_list.shape[0], J))
        for q in range(Q):
            f_j += (W[q]*true_u[:,q]).T
        #true_f.append(f_d)

        return f_j

    # True Combinations
    def W_lincombination(Q, J):
        W_list = []
        # q=1
        for q in range(Q):
            # W_list.append(np.array(([[-0.5], [0.1]])))
            if q==0:
                #W_list.append(0.3*np.random.randn(J, 1))
                W_list.append(np.array([-0.1,-0.1, 1.1, 2.1, -0.5, -0.6])[:,None])
            elif q == 1:
                #W_list.append(2.0 * np.random.randn(J, 1))
                W_list.append(np.array([1.4, -0.5, 0.3, 0.7, -0.3, 0.4])[:,None])
            else:
                #W_list.append(10.0 * np.random.randn(J, 1)+0.1)
                W_list.append(np.array([0.1, -0.8, 1.3, 1.5, -0.02, 0.01])[:,None])

        return W_list

    """"""""""""""""""""""""""""""""

    # True functions values for inputs X
    f_index = Y_metadata['function_index'].flatten()
    J = f_index.__len__()
    trueU = experiment_true_u_functions(kern_list, Xtoy)
    trueF = experiment_true_f_functions(trueU, Xtoy,J)
    # if input_dim==2:
    #     #from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    #
    #     from matplotlib import cm
    #     from matplotlib.ticker import LinearLocator, FormatStrFormatter
    #     fig = plt.figure(15)
    #     ax = fig.gca(projection='3d')
    #
    #     # Make data.
    #     # X = np.arange(-5, 5, 0.25)
    #     # Y = np.arange(-5, 5, 0.25)
    #     # X, Y = np.meshgrid(X, Y)
    #     # R = np.sqrt(X ** 2 + Y ** 2)
    #     # Z = np.sin(R)
    #
    #     # Plot the surface.
    #     surf = ax.plot_surface(Xtoy[:,0].reshape(Nsqrt,Nsqrt), Xtoy[:,1].reshape(Nsqrt,Nsqrt), trueF[:,4].reshape(Nsqrt,Nsqrt), cmap=cm.coolwarm,linewidth=0, antialiased=False)
    #
    #     # Customize the z axis.
    #     #ax.set_zlim(-1.01, 1.01)
    #     ax.zaxis.set_major_locator(LinearLocator(10))
    #     ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #
    #     # Add a color bar which maps values to colors.
    #     fig.colorbar(surf, shrink=0.5, aspect=5)
    #
    #     plt.show()
    #
    # else:
    #     plt.figure(15)
    #     plt.plot(trueF[:,4])
    #     plt.figure(16)
    #     plt.plot(trueU)

    d_index = Y_metadata['d_index'].flatten()
    F_true = []
    # for i,f_latent in enumerate(trueF):
    #    if

    for t in range(D):
        _, num_f_task, _ = likelihoods_list[t].get_metadata()
        f = np.empty((Xtoy.shape[0], num_f_task))
        for j in range(J):
            if f_index[j] == t:
                f[:, d_index[j], None] = trueF[:,j][:,None]

        F_true.append(f)

    # Generating training data Y (sampling from heterogeneous likelihood)
    Ytrain = likelihood.samples(F=F_true, Y_metadata=Y_metadata)
    #Yreg = (Ytrain[0]-Ytrain[0].min())/(Ytrain[0].max()-Ytrain[0].min())
    Yreg = (Ytrain[0] - Ytrain[0].mean(0)) / (Ytrain[0].std(0))
    Ytrain = [Yreg,np.clip(Ytrain[1],1.0e-9,0.99999),Ytrain[2]]
    Xtrain = []
    for d in range(likelihoods_list.__len__()):
        Xtrain.append(Xtoy)

    return Xtrain, Ytrain

def load_toy6(N=1000,input_dim=1):
    if input_dim==2:
        Nsqrt = int(N**(1.0/input_dim))
    print('input_dim:',input_dim)
    #Q = 5  # number of latent functions

    # Heterogeneous Likelihood Definition
    likelihoods_list = [Poisson(), Poisson()]  # Real + Binary
    likelihood = HetLikelihood(likelihoods_list)
    Y_metadata = likelihood.generate_metadata()
    D = likelihoods_list.__len__()
    Q = 2
    """"""""""""""""""""""""""""""
    rescale = 10
    Dim = input_dim
    if input_dim ==2:
        xy = rescale*np.linspace(-1.0, 1.0, Nsqrt)
        xx = rescale*np.linspace(-1.0, 1.0, Nsqrt)
        XX, XY = np.meshgrid(xx, xy)
        XX = XX.reshape(Nsqrt ** 2, 1)
        XY = XY.reshape(Nsqrt ** 2, 1)
        Xtoy = np.hstack((XX, XY))
    else:
        minis = -rescale * np.ones(Dim)
        maxis = rescale * np.ones(Dim)
        Xtoy = np.linspace(minis[0], maxis[0], N).reshape(1, -1)
        for i in range(Dim - 1):
            Xaux = np.linspace(minis[i + 1], maxis[i + 1], N)
            Xtoy = np.concatenate((Xtoy, Xaux[np.random.permutation(N)].reshape(1, -1)), axis=0)
            # Z = np.concatenate((Z, Zaux.reshape(1, -1)), axis=0)
        Xtoy = 1.0 * Xtoy.T


    def latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=None):
        if lenghtscale is None:
            lenghtscale = rescale*np.array([0.6, 0.1]) #This is the one used for previous experiments
            #lenghtscale = np.array([1.0, 0.1, 0.2])
        else:
            lenghtscale = lenghtscale

        if variance is None:
            #variance = 1.0*np.random.rand(Q)
            variance = 1 * np.ones(Q)
        else:
            variance = variance

        kern_list = []
        for q in range(Q):
            #print("length:",lenghtscale[q])
            #print("var:", variance[q])
            kern_q = GPy.kern.RBF(input_dim=input_dim, lengthscale=lenghtscale[q], variance=variance[q], name='rbf')
            kern_q.name = 'kern_q' + str(q)
            kern_list.append(kern_q)
        return kern_list

    kern_list = latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=Dim)

    # True U and F functions
    def experiment_true_u_functions(kern_list, X):
        Q = kern_list.__len__()
        u_latent = np.zeros((X.shape[0], Q))
        np.random.seed(104)
        for q in range(Q):
            u_latent[:, q] = np.random.multivariate_normal(np.zeros(X.shape[0]), kern_list[q].K(X))

        return u_latent

    def experiment_true_f_functions(true_u, X_list, J):

        Q = true_u.shape[1]
        W = W_lincombination(Q, J)
        #
        f_j = np.zeros((X_list.shape[0], J))
        for q in range(Q):
            f_j += (W[q]*true_u[:,q]).T

        return f_j

    # True Combinations
    def W_lincombination(Q, J):
        W_list = []
        for q in range(Q):
            if q==0:
                W_list.append(np.array([-0.1,2.1])[:,None])
            elif q == 1:
                W_list.append(np.array([1.4, -0.5 ])[:,None])
            else:
                W_list.append(np.array([0.1, -0.8])[:,None])

        return W_list

    """"""""""""""""""""""""""""""""

    # True functions values for inputs X
    f_index = Y_metadata['function_index'].flatten()
    J = f_index.__len__()
    trueU = experiment_true_u_functions(kern_list, Xtoy)
    trueF = experiment_true_f_functions(trueU, Xtoy,J)

    d_index = Y_metadata['d_index'].flatten()
    F_true = []


    for t in range(D):
        _, num_f_task, _ = likelihoods_list[t].get_metadata()
        f = np.empty((Xtoy.shape[0], num_f_task))
        for j in range(J):
            if f_index[j] == t:
                f[:, d_index[j], None] = trueF[:,j][:,None]

        F_true.append(f)

    # Generating training data Y (sampling from heterogeneous likelihood)
    Ytrain = likelihood.samples(F=F_true, Y_metadata=Y_metadata)
    #Yreg = (Ytrain[0] - Ytrain[0].mean(0)) / (Ytrain[0].std(0))
    #Ytrain = [Yreg,np.clip(Ytrain[1],1.0e-9,0.99999),Ytrain[2]]
    Ytrain = [Ytrain[0], Ytrain[1]]
    Xtrain = []
    for d in range(likelihoods_list.__len__()):
        Xtrain.append(Xtoy)

    return Xtrain, Ytrain


def load_Hetdata(dataset='london', Ntoy= None, Only_input_dim_toy = None, Ngrid_sel = None, Nday = None):
    if (dataset=='toy1' or dataset=='toy2' or dataset=='toy3'or dataset=='toy4'or dataset=='toy5') and ((Only_input_dim_toy is None) or (Ntoy is None)):
        print('For the toys you have to provide second and third arguments as Ndata and input dim respectively!')
    elif dataset=='toy1':
        Xtrain , Ytrain = load_toy1(N=Ntoy,input_dim=Only_input_dim_toy)
    elif dataset=='toy1c':
        Xtrain , Ytrain = load_toy1_conv(N=Ntoy,input_dim=Only_input_dim_toy)
    elif dataset=='toy2c':
        Xtrain , Ytrain = load_toy2_conv(N=Ntoy,input_dim=Only_input_dim_toy)
    elif dataset=='toy3c':
        Xtrain , Ytrain = load_toy3_conv(N=Ntoy,input_dim=Only_input_dim_toy)
    elif dataset=='toy2':
        Xtrain, Ytrain = load_toy2(N=Ntoy, input_dim=Only_input_dim_toy)
    elif dataset=='toy3':
        Xtrain, Ytrain = load_toy3(N=Ntoy, input_dim=Only_input_dim_toy)
    elif dataset=='toy4':
        Xtrain, Ytrain = load_toy4(N=Ntoy, input_dim=Only_input_dim_toy)
    elif dataset=='toy5':
        Xtrain, Ytrain = load_toy5(N=Ntoy, input_dim=Only_input_dim_toy)
    elif dataset=='toy6':
        Xtrain, Ytrain = load_toy6(N=Ntoy, input_dim=Only_input_dim_toy)
    elif dataset=='bern':
        Xtrain, Ytrain = load_bern(N=Ntoy, input_dim=Only_input_dim_toy)
    elif dataset=='zip':
        Xtrain, Ytrain = load_ZIP(N=Ntoy, input_dim=Only_input_dim_toy)
    elif dataset=='zip_conv1':
        Xtrain, Ytrain = load_ZIP_conv1(N=Ntoy, input_dim=Only_input_dim_toy)
    elif dataset=='zip_conv2':
        Xtrain, Ytrain = load_ZIP_conv2(N=Ntoy, input_dim=Only_input_dim_toy)
    elif dataset=='zip_conv3':
        Xtrain, Ytrain = load_ZIP_conv3(N=Ntoy, input_dim=Only_input_dim_toy)
    elif dataset=='zip_conv4':
        Xtrain, Ytrain = load_ZIP_conv4(N=Ntoy, input_dim=Only_input_dim_toy)
    elif dataset=='nbinomial_conv1':
        Xtrain, Ytrain = load_NB_conv1(N=Ntoy, input_dim=Only_input_dim_toy)

    else:
        print("The dataset doesn't exist!")
        return 0

    return Xtrain, Ytrain

