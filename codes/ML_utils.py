"""
Utility tools for the ML-based metric learning scheme
"""

# Understanding the metric learning stage

import numpy as np
import matplotlib.pyplot as plt
import IAE_JAX as mldr

fsize = 16
vcol = ['mediumseagreen', 'crimson', 'steelblue', 'darkmagenta', 'burlywood', 'khaki', 'lightblue', 'darkseagreen',
        'deepskyblue', 'forestgreen', 'gold', 'indianred', 'midnightblue', 'olive', 'orangered', 'orchid', 'red',
        'steelblue']
font = {'family': 'normal', 'weight': 'bold', 'size': fsize}
plt.rc('font', **font)


def plot_histo(H):
    u = [(a + b) / 2 for a, b in zip(H[1], H[1][1::1])]
    plt.plot(u, H[0])


# Comparison metrics

def ComparisonMetrics_SampleDomain(X, model=None, fname=None, noise_level=None, relative=False, p=25, P=75, res=True):
    """
    Comparison metrics in the sample domain
    """

    if fname is not None:
        model = mld.load_model(fname)

    if res:
        LearnFunc = mldr.MetricLearning(Params=model)
    else:
        LearnFunc = mld.MetricLearning(Params=model)

    if noise_level is not None:
        noise = np.random.randn(X.shape[0], X.shape[1])
        Xtest = X + np.power(10., -noise_level / 20) * noise / np.linalg.norm(noise) * np.linalg.norm(X)
        Output = LearnFunc.fast_interpolation(Xtest)
    else:
        Output = LearnFunc.fast_interpolation(X)

    Xrec = Output["Xrec"]

    if relative:
        Error = abs(X - Xrec) / (1e-16 + X)
    else:
        Error = abs(X - Xrec)

    mse = np.sqrt(np.sum(Error ** 2, axis=1) / Error.shape[1])
    me = np.mean(Error, axis=1)
    med = np.median(Error, axis=1)

    plt.figure(figsize=[12, 10])
    n, bins, patches = plt.hist(-20. * np.log10(mse), 50, density=False, facecolor=vcol[0], alpha=0.75)
    plt.xlabel("MSE in dB")
    plt.ylabel("Histogram")
    plt.savefig(fname + '_MSE_SampleDomain.tif')

    return {"vMSE": mse, "vMean": me, "vMedian": med, "pMSE": np.percentile(mse, p), "pMean": np.percentile(me, p),
            "pMedian": np.percentile(med, p), "PMSE": np.percentile(mse, P), "PMean": np.percentile(me, P),
            "PMedian": np.percentile(med, P), "mMSE": np.percentile(mse, 50), "mMean": np.percentile(me, 50),
            "mMedian": np.percentile(med, 50)}


def ComparisonMetrics_TransDomain(X, model=None, fname=None, noise_level=None, Simplex=None, relative=False, p=25, P=75,
                                  res=True):
    """
    Comparison metrics in the transformed domain
    """

    if fname is not None:
        model = mld.load_model(fname)

    if res:
        LearnFunc = mldr.MetricLearning(Params=model)
    else:
        LearnFunc = mld.MetricLearning(Params=model)

    if noise_level is not None:
        noise = np.random.randn(X.shape[0], X.shape[1])
        Xtest = X + np.power(10., -noise_level / 20) * noise / np.linalg.norm(noise) * np.linalg.norm(X)
        Output = LearnFunc.fast_interpolation(Xtest, simplex=Simplex)

    else:
        Output = LearnFunc.fast_interpolation(X, simplex=Simplex)

    phiX = Output["phiX"]
    B = Output["Barycenter"]

    if relative:
        Error = abs(phiX - B) / (1e-16 + phiX)  # impact of noise onto the barycenter
    else:
        Error = abs(phiX - B)

    mse = np.sqrt(np.sum(Error ** 2, axis=1) / Error.shape[1])
    me = np.mean(Error, axis=1)
    med = np.median(Error, axis=1)

    plt.figure(figsize=[12, 10])
    n, bins, patches = plt.hist(-20. * np.log10(mse), 50, density=False, facecolor=vcol[0], alpha=0.75)
    plt.xlabel("MSE in dB")
    plt.ylabel("Histogram")
    plt.savefig(fname + '_MSE_CodeDomain.tif')

    return {"vMSE": mse, "vMean": me, "vMedian": med, "pMSE": np.percentile(mse, p), "pMean": np.percentile(me, p),
            "pMedian": np.percentile(med, p), "PMSE": np.percentile(mse, P), "PMean": np.percentile(me, P),
            "PMedian": np.percentile(med, P), "mMSE": np.percentile(mse, 50), "mMean": np.percentile(me, 50),
            "mMedian": np.percentile(med, 50)}


def ComparisonMetrics_TransDomain_Barycenters(X, model=None, fname=None, noise_level=None, Simplex=None, relative=False,
                                              p=25, P=75, res=True):
    """
    Comparison metrics in the transformed domain
    """

    if fname is not None:
        model = mld.load_model(fname)

    if res:
        LearnFunc = mldr.MetricLearning(Params=model)
    else:
        LearnFunc = mld.MetricLearning(Params=model)

    if noise_level is not None:
        noise = np.random.randn(X.shape[0], X.shape[1])
        Xtest = X + np.power(10., -noise_level / 20) * noise / np.linalg.norm(noise) * np.linalg.norm(X)
        OutputN = LearnFunc.fast_interpolation(Xtest)
        Output = LearnFunc.fast_interpolation(X, simplex=Simplex)
        phiX = Output["phiX"]
        B = Output["Barycenter"]
        Bn = OutputN["Barycenter"]

        if relative:
            Error = abs(Bn - B) / (1e-16 + B)  # impact of noise onto the barycenter
        else:
            Error = abs(Bn - B)
    else:
        Output = LearnFunc.fast_interpolation(X, simplex=Simplex)
        phiX = Output["phiX"]
        B = Output["Barycenter"]

        if relative:
            Error = abs(phiX - B) / (1e-16 + X)
        else:
            Error = abs(phiX - B)

    mse = np.sqrt(np.sum(Error ** 2, axis=1) / Error.shape[1])
    me = np.mean(Error, axis=1)
    med = np.median(Error, axis=1)

    return {"vMSE": mse, "vMean": me, "vMedian": med, "pMSE": np.percentile(mse, p), "pMean": np.percentile(me, p),
            "pMedian": np.percentile(med, p), "PMSE": np.percentile(mse, P), "PMean": np.percentile(me, P),
            "PMedian": np.percentile(med, P), "mMSE": np.percentile(mse, 50), "mMean": np.percentile(me, 50),
            "mMedian": np.percentile(med, 50)}


def ComparisonMetrics_TransDomain(X, model=None, fname=None, noise_level=None, Simplex=None, relative=False, p=25, P=75,
                                  res=True):
    """
    Comparison metrics in the transformed domain
    """

    if fname is not None:
        model = mld.load_model(fname)

    if res:
        LearnFunc = mldr.MetricLearning(Params=model)
    else:
        LearnFunc = mld.MetricLearning(Params=model)

    if noise_level is not None:
        noise = np.random.randn(X.shape[0], X.shape[1])
        Xtest = X + np.power(10., -noise_level / 20) * noise / np.linalg.norm(noise) * np.linalg.norm(X)
        Output = LearnFunc.fast_interpolation(Xtest, simplex=Simplex)

    else:
        Output = LearnFunc.fast_interpolation(X, simplex=Simplex)

    phiX = Output["phiX"]
    B = Output["Barycenter"]

    if relative:
        Error = abs(phiX - B) / (1e-16 + phiX)  # impact of noise onto the barycenter
    else:
        Error = abs(phiX - B)

    mse = np.sqrt(np.sum(Error ** 2, axis=1) / Error.shape[1])
    me = np.mean(Error, axis=1)
    med = np.median(Error, axis=1)

    plt.figure(figsize=[12, 10])
    n, bins, patches = plt.hist(-20. * np.log10(mse), 50, density=False, facecolor=vcol[0], alpha=0.75)
    plt.xlabel("MSE in dB")
    plt.ylabel("Histogram")
    plt.savefig(fname + '_MSE_CodeDomain.tif')

    return {"vMSE": mse, "vMean": me, "vMedian": med, "pMSE": np.percentile(mse, p), "pMean": np.percentile(me, p),
            "pMedian": np.percentile(med, p), "PMSE": np.percentile(mse, P), "PMean": np.percentile(me, P),
            "PMedian": np.percentile(med, P), "mMSE": np.percentile(mse, 50), "mMean": np.percentile(me, 50),
            "mMedian": np.percentile(med, 50)}


def ComparisonMetrics_BarycentricSpanProjection(X, model=None, fname=None, noise_level=None, Simplex=1, alpha=0.5,
                                                label=None, relative=False, p=25, P=75, res=True, col=0, savefig=False,
                                                newfig=False, PlotScatter=False):
    """
    Comparison metrics in the transformed domain
    """

    if fname is not None:
        model = mld.load_model(fname)

    if res:
        LearnFunc = mldr.MetricLearning(Params=model)
    else:
        LearnFunc = mld.MetricLearning(Params=model)

    if noise_level is not None:
        noise = np.random.randn(X.shape[0], X.shape[1])
        Xtest = X + np.power(10., -noise_level / 20) * noise / np.linalg.norm(noise) * np.linalg.norm(X)
        Output = LearnFunc.barycentric_span_projection(Xtest)

    else:
        Output = LearnFunc.barycentric_span_projection(X)

    Xrec = Output["Xrec"]

    if relative:
        Error = abs(X - Xrec) / (1e-16 + X)
    else:
        Error = abs(X - Xrec)

    mse = np.sqrt(np.sum(Error ** 2, axis=1) / Error.shape[1])
    me = np.mean(Error, axis=1)
    med = np.median(Error, axis=1)

    if newfig:
        plt.figure(figsize=[12, 10])
    n, bins, patches = plt.hist(-20. * np.log10(mse), bins='auto', density=True, facecolor=vcol[col], alpha=alpha)
    plt.xlabel("MSE in dB")
    plt.ylabel("Histogram")
    if label is not None:
        plt.legend(label)
    plt.title('median MSE %2.2E' % np.percentile(-20. * np.log10(mse), 50) + ' - p25 : %2.2E' % np.percentile(
        -20. * np.log10(mse), 25) + ' - p75 : %2.2E' % np.percentile(-20. * np.log10(mse), 75))
    if savefig:
        plt.savefig(fname + '_MSE_CodeDomain_MSE.tif')

    if PlotScatter:
        PlotInterp_SampleDomain(X, sval=200, cval=-20. * np.log10(mse), fname=fname, savefig=savefig)
        PlotInterp_TransPhi(X, cval=-20. * np.log10(mse), sval=200, fname=fname, savefig=savefig)

    return {"vMSE": mse, "vMean": me, "vMedian": med, "pMSE": np.percentile(mse, p), "pMean": np.percentile(me, p),
            "pMedian": np.percentile(med, p), "PMSE": np.percentile(mse, P), "PMean": np.percentile(me, P),
            "PMedian": np.percentile(med, P), "mMSE": np.percentile(mse, 50), "mMean": np.percentile(me, 50),
            "mMedian": np.percentile(med, 50)}


def PlotInterp_TransPhi(X, model=None, noise_level=None, fname=None, cval=None, sval=None, Simplex=None, res=True,
                        savefig=False):
    """
    Plotting the interpolation in the direct domain
    """

    if fname is not None:
        model = mld.load_model(fname)

    if res:
        LearnFunc = mldr.MetricLearning(Params=model)
    else:
        LearnFunc = mld.MetricLearning(Params=model)

    if noise_level is not None:
        noise = np.random.randn(X.shape[0], X.shape[1])
        Xtest = X + np.power(10., -noise_level / 20) * noise / np.linalg.norm(noise) * np.linalg.norm(X)
        Output = LearnFunc.fast_interpolation(Xtest, simplex=Simplex)
        phiX = Output["phiX"]
        B = Output["Barycenter"]
        Output = LearnFunc.fast_interpolation(X, simplex=Simplex)
        phiXt = Output["phiX"]
        Bt = Output["Barycenter"]
        phiE = Output["phiE"]
        R = phiE.T @ phiE
        U, _, _ = np.linalg.svd(R)
        pX = phiX @ U
        pXt = phiXt @ U
        pE = phiE @ U
        pB = B @ U
        pBt = Bt @ U
        plt.figure(figsize=[16, 10])
        plt.scatter(pE[:, 0], pE[:, 1], marker='P', s=500, cmap='Reds', alpha=0.5)
        plt.scatter(pX[:, 0], pX[:, 1], marker='o', c=cval, s=300, cmap='Greens', alpha=0.2)
        plt.scatter(pB[:, 0], pB[:, 1], marker='s', c=cval, s=300, cmap='Purples', alpha=0.2)
        plt.scatter(pXt[:, 0], pXt[:, 1], marker='o', c=cval, s=150, cmap='Yellows', alpha=0.2)
        plt.scatter(pBt[:, 0], pBt[:, 1], marker='s', c=cval, s=150, cmap='Blues', alpha=0.2)

    else:
        plt.rcParams.update({'font.size': 18})
        Output = LearnFunc.fast_interpolation(X, simplex=Simplex)
        phiX = Output["phiX"]
        phiE = Output["phiE"]
        B = Output["Barycenter"]
        R = phiE.T @ phiE
        U, _, _ = np.linalg.svd(R)
        pX = phiX @ U
        pE = phiE @ U
        pB = B @ U
        plt.figure(figsize=[12, 10])
        plt.scatter(pE[:, 0], pE[:, 1], marker='P', s=2 * sval, cmap='Reds', alpha=0.5)
        plt.scatter(pX[:, 0], pX[:, 1], marker='x', c=cval, s=sval, cmap='Greens', alpha=0.2)
        plt.scatter(pB[:, 0], pB[:, 1], marker='s', c=cval, s=sval, cmap='Purples', alpha=0.2)
        plt.legend(["Anchor Points", "Input", "Interpolated"])
        plt.colorbar()

    if savefig:
        plt.savefig(fname + '_Scatter_CodeDomain.tif')


def PlotInterp_SampleDomain(X, model=None, noise_level=None, fname=None, cval=None, sval=None, Simplex=None, res=True,
                            savefig=False):
    """
    Plotting the interpolation in the direct domain
    """

    if fname is not None:
        model = mld.load_model(fname)

    if res:
        LearnFunc = mldr.MetricLearning(Params=model)
    else:
        LearnFunc = mld.MetricLearning(Params=model)

    if noise_level is not None:
        noise = np.random.randn(X.shape[0], X.shape[1])
        Xtest = X + np.power(10., -noise_level / 20) * noise / np.linalg.norm(noise) * np.linalg.norm(X)
        Output = LearnFunc.fast_interpolation(Xtest, simplex=Simplex)
        phiX = Output["phiX"]
        B = Output["Xrec"]
        Output = LearnFunc.fast_interpolation(X, simplex=Simplex)
        phiXt = Output["phiX"]
        Bt = Output["Xrec"]
        phiE = Output["phiE"]
        R = X.T @ X
        U, _, _ = np.linalg.svd(R)
        pX = X @ U
        pXt = Xtest @ U
        pE = model["AnchorPoints"] @ U
        pB = B @ U
        pBt = Bt @ U
        plt.figure(figsize=[16, 10])
        plt.scatter(pE[:, 0], pE[:, 1], marker='P', s=500, cmap='Reds', alpha=0.5)
        plt.scatter(pX[:, 0], pX[:, 1], marker='o', c=cval, s=300, cmap='Greens', alpha=0.2)
        # plt.scatter(pB[:,0],pB[:,1],marker='s',c=cval,s=300,cmap='Purples',alpha=0.2)
        plt.scatter(pXt[:, 0], pXt[:, 1], marker='o', c=cval, s=150, cmap='Yellows', alpha=0.2)
        # plt.scatter(pBt[:,0],pBt[:,1],marker='s',c=cval,s=150,cmap='Blues',alpha=0.2)

    else:
        plt.rcParams.update({'font.size': 18})
        Output = LearnFunc.fast_interpolation(X, simplex=Simplex)
        R = X.T @ X
        U, _, _ = np.linalg.svd(R)
        pX = X @ U
        pE = model["AnchorPoints"] @ U
        pB = Output["Xrec"] @ U
        plt.figure(figsize=[12, 10])
        plt.scatter(pE[:, 0], pE[:, 1], marker='P', s=2 * sval, cmap='Reds', alpha=0.75)
        plt.scatter(pX[:, 0], pX[:, 1], marker='x', c=cval, s=sval, cmap='Greens', alpha=0.2)
        plt.scatter(pB[:, 0], pB[:, 1], marker='s', c=cval, s=sval, cmap='Purples', alpha=0.2)
        plt.legend(["Anchor Points", "Input", "Interpolated"])
        plt.colorbar()

    if savefig:
        plt.savefig(fname + '_Scatter_SampleDomain.tif')


def Get_MSEvsNoiseSNR(Xin, fname, snr_min=10, snr_max=40, nval=10, nsamples=10, res=True):
    ampp = 1.
    model = mld.load_model(fname)
    if res:
        LearnFunc = mldr.MetricLearning(Params=model)
    else:
        LearnFunc = mld.MetricLearning(Params=model)
    vSNR = np.linspace(snr_min, snr_max, nval)
    MSE = []
    MSEls = []

    for snr in vSNR:
        Xb = ampp * Xin + (10. ** (-snr / 20.)) * np.random.randn(Xin.shape[0], Xin.shape[1])
        Outputt = LearnFunc.fast_interpolation(Xb[0:nsamples, :])
        MSE.append(np.linalg.norm(Outputt["Xrec"] - Xin[0:nsamples, :]))
        outt = LearnFunc.barycentric_span_projection(Xb[0:nsamples, :])
        MSEls.append(np.linalg.norm(outt["Rec"] - Xin[0:nsamples, :]))

    return MSE, MSEls, vSNR
