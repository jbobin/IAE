"""

Pieces of codes to generate manifold-valued data

This (will) include(s):
    - Thermal/synchrotron Chandra spectra
    - Gaussian-shaped samples
    - MNIST samples

Version : 0.1 - April, 20th 2020
Author : J.Bobin - CEA

"""

import numpy as np


def GaussianShaped(t_length=44, t_start=20, pmin=15, pmax=35, train_samp_number=150, validation_samp_number=150,
                   test_samp_number=150, law='linear', freq=1, dim=None, complex_output=False):
    """
    Gaussian-shaped samples
    """

    output = {}

    def GaussSpec(t=25, wmin=0.03, wmax=0.08, p=5, law=law):

        if law == 'linear':
            w = wmax * p / t + wmin * (t - p) / t
        if law == 'quadratic':
            w = wmax * p ** 2. / t ** 2. + wmin * (t ** 2. - p ** 2.) / t ** 2.
        if law == 'periodic':
            w = (wmax - wmin) / 2 * (np.sin(freq * np.pi * p / t) + 1) + wmin
        g = np.linspace(0, t, t)
        g = np.exp(-w * (g - p) ** 2)
        g = g / np.sum(g)

        return g

    # With 2 examples only

    Psi = np.zeros((8, t_length))
    p_Psi = np.array([15, 35, 20, 30, 18, 32, 24, 28])
    for r in range(8):
        Psi[r, :] = GaussSpec(t=t_length, p=p_Psi[r], wmax=0.1, wmin=0.01)
    if dim is not None:
        Psi = np.repeat(Psi[:, :, np.newaxis], dim, axis=2)

    # Training set

    X_train = np.zeros((train_samp_number, t_length))
    p_train = (pmax - pmin) * np.random.rand(train_samp_number) + pmin

    for r in range(train_samp_number):
        X_train[r, :] = GaussSpec(t=t_length, p=p_train[r], wmax=0.1, wmin=0.01)
    if dim is not None:
        X_train = np.repeat(X_train[:, :, np.newaxis], dim, axis=2)

    output["X_train"] = X_train
    output["p_train"] = p_train

    # Test set

    X_test = np.zeros((test_samp_number, t_length))
    p_test = (pmax - pmin) * np.random.rand(train_samp_number) + pmin

    for r in range(test_samp_number):
        X_test[r, :] = GaussSpec(t=t_length, p=p_test[r], wmax=0.1, wmin=0.01)
    if dim is not None:
        X_test = np.repeat(X_test[:, :, np.newaxis], dim, axis=2)

    output["X_test"] = X_test
    output["p_test"] = p_test

    # Validation set

    X_valid = np.zeros((validation_samp_number, t_length))
    p_valid = (pmax - pmin) * np.random.rand(validation_samp_number) + pmin

    for r in range(validation_samp_number):
        X_valid[r, :] = GaussSpec(t=t_length, p=p_valid[r], wmax=0.1, wmin=0.01)
    if dim is not None:
        X_valid = np.repeat(X_valid[:, :, np.newaxis], dim, axis=2)

    output["X_valid"] = X_valid
    output["p_valid"] = p_valid

    output["Psi"] = Psi

    if (dim is None or dim == 1) and complex_output:
        angles = np.exp(2*np.pi*1j*np.linspace(0, 1, t_length+1)[:-1])[np.newaxis, :]
        output["X_train"] = output["X_train"].squeeze()*angles
        output["X_test"] = output["X_test"].squeeze()*angles
        output["X_valid"] = output["X_valid"].squeeze()*angles
        output["Psi"] = output["Psi"].squeeze()*angles

    return output


def Binary(t_length=8, train_samp_number=150, validation_samp_number=150, test_samp_number=150):
    def GenerateSingleBinary(nsize, b, sl, ud):
        if ud > 0:
            u = np.zeros((nsize, nsize))
            c = 1
        else:
            u = np.ones((nsize, nsize))
            c = 0

        for rx in range(nsize):
            tlim = np.int(sl * rx + b)
            if tlim > nsize:
                tlim = nsize
            if tlim > -1:
                u[rx, 0:tlim] = c

        return u

    output = {}

    b = t_length * (np.random.rand(train_samp_number, ) / 2 + 0.5)
    slope = 5. * (np.random.rand(train_samp_number, ) - 0.5)
    UpDownBW = np.sign(np.random.randn(train_samp_number, ) - 0.5)
    print(np.min(slope), np.max(slope))
    X_train = np.zeros((train_samp_number, t_length, t_length))

    for r in range(train_samp_number):
        X_train[r, :, :] = GenerateSingleBinary(t_length, b[r], slope[r], UpDownBW[r])

    output["X_train"] = X_train
    output["b_train"] = b
    output["UpDownBW_train"] = UpDownBW
    output["slope_train"] = slope

    b = t_length * (np.random.rand(train_samp_number, ) / 2 + 0.5)
    slope = 5. * (np.random.rand(train_samp_number, ) - 0.5)
    UpDownBW = np.sign(np.random.randn(train_samp_number, ) - 0.5)
    print(np.min(slope), np.max(slope))
    X_test = np.zeros((test_samp_number, t_length, t_length))

    for r in range(train_samp_number):
        X_test[r, :, :] = GenerateSingleBinary(t_length, b[r], slope[r], UpDownBW[r])

    output["X_test"] = X_test
    output["b_test"] = b
    output["UpDownBW_test"] = UpDownBW
    output["slope_test"] = slope

    b = t_length * (np.random.rand(train_samp_number, ) / 2 + 0.5)
    slope = 5. * (np.random.rand(train_samp_number, ) - 0.5)
    UpDownBW = np.sign(np.random.randn(train_samp_number, ) - 0.5)
    Psi = np.zeros((16, t_length, t_length))
    for r in range(16):
        Psi[r, :, :] = GenerateSingleBinary(t_length, b[r], slope[r], UpDownBW[r])

    output["Psi"] = Psi

    return output
