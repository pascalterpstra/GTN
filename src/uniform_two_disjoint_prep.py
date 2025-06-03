import matplotlib.pyplot as plt
import torch
from utils import make_dirs, save_dataset_as_torch

def make_uniform_2d_dataset(dir_dataset, n_samples):

    dir_out = dir_dataset
    make_dirs([dir_out])

    size_train, size_val, size_test = n_samples, n_samples//5, n_samples//5  # to mimic the usual 20% val, test

    import numpy as np
    sign_plus = np.ones(size_train//2)
    sign_minus = -1*sign_plus
    sign_ = np.concatenate([sign_plus, sign_minus])
    np.random.shuffle(sign_)
    sign = [[-1,-1] if i == -1 else [1,1] for i in sign_]
    addition = [[-0.25, -0.25] if i == -1 else [0.25,0.25] for i in sign_]
    sign = torch.Tensor(sign)
    addition = torch.Tensor(addition)

    train = torch.rand(size=(size_train, 2))*sign+addition
    val = torch.rand(size=(size_val, 2))*sign[:size_val]+addition[:size_val]
    test = torch.rand(size=(size_test, 2))*sign[:size_test]+addition[:size_test]

    # plt.plot(train[:,0], train[:,1])
    # plt.show()

    save_dataset_as_torch(train, val, test, dir_out, scale_01=False)

if __name__ == '__main__':
    print("Preparing the data.")
    make_uniform_2d_dataset('../data/2d_two_disjoint_uniforms/', 100000)
    print("Done.")