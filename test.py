from vdpgmm import VDPGMM
from sklearn import preprocessing
from sklearn import datasets
import matplotlib.pylab as plt
import matplotlib as mpl
import numpy as np
import scipy as sp

def getXY(dataset = 'boston'):
    X = None
    Y = None
    if dataset == 'boston':
        boston = datasets.load_boston()
        X = boston.data
        Y = boston.target
    elif dataset == 'diabetes':
        ds = datasets.load_diabetes()
        X = ds.data
        Y = ds.target
    elif dataset == 'iris':
        ds = datasets.load_iris()
        X = ds.data
        Y = ds.target
    elif dataset == 'digits':
        ds = datasets.load_digits()
        X = ds.data
        Y = ds.target
    return X, Y

def test1():
    print('test1')
    model = VDPGMM(T = 10, alpha = 1, max_iter = 50)
    X, Y = getXY('iris')
    print(X.shape)
    print(X[:10])

    model.fit(X)
    y = model.predict(X)
    print('VDPGMM')
    print(len(np.unique(y)), np.unique(y))
    print([np.sum(y == label) for label in np.unique(y)])

    # from sklearn.mixture import DPGMM
    from sklearn.mixture import BayesianGaussianMixture
    model = BayesianGaussianMixture(n_components = 10, weight_concentration_prior = 1, max_iter = 1000)
    model.fit(X)
    y = model.predict(X)
    print('DPGMM')
    print(len(np.unique(y)), np.unique(y))
    print([np.sum(y == label) for label in np.unique(y)])

def test2():
    print('test2')
    np.random.seed(1)
    X = np.concatenate((2 + np.random.randn(100, 2), 5 + np.random.randn(100, 2),  10 + np.random.randn(100, 2)))
    T = 10
    model = VDPGMM(T=T, alpha=.5, max_iter=100, thresh=1e-5)
    model.fit(X)
    
    plt.clf()
    h = plt.subplot()
    color = 'rgbcmykw'
    k = 0
    clusters = np.argmax(model.phi, axis=0)
    for t in range(T):
            xt = X[clusters == t, :]
            nt = xt.shape[0]
            if nt != 0:
                print(t, nt, model.mean_mu[t,:])
                ell = mpl.patches.Ellipse(model.mean_mu[t,:], 1, 1, 50, color = color[k])
                ell.set_alpha(0.5)
                plt.scatter(xt[:, 0], xt[:, 1], color = color[k])
                h.add_artist(ell)
                k += 1

    plt.show()


def _main():
    # test1()

    # test2()

    # load cifar10 image datasets
    import pickle
    file = "./cifar-10-batches-py/data_batch_1"
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')

    # I only use 100 pics to train, because my laptop is uhuh, can try larger data sets later
    # X = dict[b'data'][:5000,:]
    X = dict[b'data']

    import matplotlib.pyplot as plt
    # print(X.shape)

    model = VDPGMM(T = 20, alpha = 1, max_iter = 5000)
    model.fit(X)
    y = model.predict(X)
    print('VDPGMM')
    print(len(np.unique(y)), np.unique(y))
    count = [np.sum(y == label) for label in np.unique(y)]
    print(count)

    # save pics index from each cluster
    index = []
    for label in np.unique(y):
        index.append([i for i in range(len(y)) if y[i] == label])

    # plot 10 pics for each cluster... 
    fig, axes1 = plt.subplots(len(np.unique(y)),10,figsize=(3,3))
    for i in range(len(index)):
        X_ = X[index[i]][:10]
        X_ = X_.reshape(len(X_),3,32,32).transpose(0,2,3,1).astype("uint8")
        for j in range(len(X_)):
            axes1[i][j].set_axis_off()
            axes1[i][j].imshow(X_[j:j+1][0])
    plt.show()


if __name__ == '__main__':
    _main()