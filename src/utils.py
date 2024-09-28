# Copyright 2024 authors of the paper "Generative Topological Networks".
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

print("Importing packages: torch, numpy torchvision etc. If this is the first run this could take about a minute.")
from torch.utils.data import Dataset
import numpy as np
from conf import *
from matplotlib import pyplot as plt
from scipy.stats import norm
import pickle as pkl
import torchvision.utils
from PIL import Image
import math
from tqdm import tqdm
import os
import pandas as pd
print("Imports complete.")


phi, phi_inv = norm.cdf, norm.ppf
np.random.seed(random_seed)
torch.manual_seed(random_seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


def savePkl(data, dir, filename):
    if not os.path.exists(dir):
        os.mkdir(dir)
    handler = open(dir+filename, 'wb')
    pkl.dump(data, handler)
    handler.close()


def loadPkl(dir, filename):
    handler = open(dir+filename, 'rb')
    data = pkl.load(handler)
    handler.close()
    return data


def swiss_roll(theta):
    return theta * [math.cos(i) for i in theta], theta * [math.sin(i) for i in theta]


def save_dataset_as_torch(train, val, test, dir_dataset, scale_01=False):
    if scale_01:
        train_mean, train_std = torch.mean(train, axis=0), torch.std(train, axis=0)
    else:
        train_mean, train_std = 0, 1

    # Normalize using train stats
    train = (train - train_mean) / train_std
    val = (val - train_mean) / train_std
    test = (test - train_mean) / train_std

    train = train.type(torch.float32)
    val = val.type(torch.float32)
    test = test.type(torch.float32)

    if not os.path.exists(dir_dataset):
        os.makedirs(dir_dataset)

    torch.save(train, dir_dataset + 'train.pt')
    torch.save(val, dir_dataset + 'val.pt')
    torch.save(test, dir_dataset + 'test.pt')

    torch.save(train_mean, dir_dataset+'train_mean.pt')
    torch.save(train_std, dir_dataset+'train_std.pt')


def calc_normalization_stats_from_train(path_train, dir_dataset):

    data_train = torch.load(path_train, map_location=torch.device(DEVICE))

    train_mean, train_std = torch.mean(data_train, dim=0), torch.std(data_train, dim=0)

    torch.save(train_mean, dir_dataset+'train_mean.pt')
    torch.save(train_std, dir_dataset+'train_std.pt')

    return train_mean, train_std


def create_single_image_from_tensor(img_tensor, reverse_normalization=True):
    if reverse_normalization:
        img_tensor = 0.5 * img_tensor + 0.5
        img_tensor = img_tensor.clamp(0, 1)

    img_tensor = img_tensor.numpy()
    img_tensor = np.transpose(img_tensor, (1, 2, 0))
    return img_tensor


def save_images_grid(data, out_folder, filename, save=False, reverse_normalization=True, nrow=10, dpi=2000):
    print("Plotting using dpi={}. This may be slow for high dpi and may take up extra space.".format(dpi))
    fig, ax = plt.subplots(figsize=(5, 5))
    img = create_single_image_from_tensor(torchvision.utils.make_grid(data, nrow, 5), reverse_normalization)
    plt.imshow(img)
    plt.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False)
    plt.tick_params(
        axis='y',
        which='both',
        left=False,
        right=False,
        labelleft=False,
        labelright=False)
    plt.axis('off')
    plt.savefig(out_folder + filename, dpi=dpi)
    plt.close()


def calc_normalization_stats_from_train(path_train, dir_dataset):

    data_train = torch.load(path_train, map_location=torch.device(DEVICE))

    train_mean, train_std = torch.mean(data_train, dim=0), torch.std(data_train, dim=0)

    torch.save(train_mean, dir_dataset+'train_mean.pt')
    torch.save(train_std, dir_dataset+'train_std.pt')

    return train_mean, train_std


class NormalToTrgtDataset(Dataset):
    '''
    Creates a dataset object that produces (y, x_y) pairs where y is sampled from the standard normal distribution and
    x_y is from the dataset (see paper).  The data samples x_y are assumed to be roughly 0-centred. Note that below,
    x and y represent input and labels respectively, for consistency with typical training notation (so x below is
    actually the y from the pair (y, x_y) and y below is actually the x_y from the pair (y, x_y).
    '''
    def __init__(self, trgt_filepath, dataset_path, subset, transform=None, target_transform=None, type="cosine"):
        self.trgt = torch.load(trgt_filepath, map_location=torch.device(DEVICE))
        # Choosing Gaussian as the source distribution (Gaussian):
        self.x = torch.randn(self.trgt.shape, device=DEVICE)

        if 'x_{}.pt'.format(subset) in os.listdir(dataset_path) and 'y_{}.pt'.format(subset) in os.listdir(dataset_path):
            x_filepath = dataset_path + 'x_{}.pt'.format(subset)
            y_filepath = dataset_path + 'y_{}.pt'.format(subset)
            print("Found dataset with cosine sim labels: {} and {}. Loading.".format(x_filepath, y_filepath))
            self.x = torch.load(x_filepath, map_location=torch.device(DEVICE))
            self.y = torch.load(y_filepath, map_location=torch.device(DEVICE))
        else:
            self.x, self.y = create_labels(self.x, self.trgt, type=type)
            torch.save(self.x, dataset_path + 'x_{}.pt'.format(subset))
            torch.save(self.y, dataset_path + 'y_{}.pt'.format(subset))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y


class CelebaDataset(Dataset):

    def __init__(self, csv_path, img_dir, attribute_name=None, transform=None):
        df = pd.read_csv(csv_path, index_col=0)
        self.df_attr = None
        self.img_dir = img_dir
        self.csv_path = csv_path
        if attribute_name is not None: # if you wish to use list_attr_celeba.csv (see here below) -- you can downloaded it from Kaggle since the original file is faulty: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset?select=list_attr_celeba.csv
            self.df_attr = pd.read_csv('../res/list_attr_celeba.csv', index_col=0)
            self.df_attr = self.df_attr.loc[self.df_attr[attribute_name] == 1]
            ix_attr = df.index.intersection(self.df_attr.index)
            df = df.loc[ix_attr]
        self.img_names = df.index.values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor([0])

    def __len__(self):
        return len(self.img_names)


class HandsDataset(Dataset):

    def __init__(self, csv_path, img_dir, transform=None):
        df = pd.read_csv(csv_path, index_col=None)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df['Filename']
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      str(self.img_names[index])))

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor([0])

    def __len__(self):
        return len(self.img_names)


class EncodedImagesDataset(Dataset):
    def __init__(self, trgt_filepath, transform=None, target_transform=None):
        self.x = torch.load(trgt_filepath, map_location=torch.device(DEVICE))
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform:
            x = self.transform(x)
        return x


def create_labels(s, t, type="cosine"):
    '''
    Creates the labeled pairs (y, x_y) as described in the paper.
    :param s: tensor of shape n_samples  x  d (d = latent dimension, see paper)
                representing source distribution (Y in paper) to map to target t (next param)
    :param t: tensor of shape n_samples  x  d (d = latent dimension, see paper) representing target (X in paper)
    :return: source s with matching labels from target
    '''
    print("Creating labeled data.")
    assert (type == "cosine") or (type == "angular_dist"), "You must choose as 'type' one of: 'cosine' or 'angular_dist'. "

    t_norm, s_norm = torch.linalg.norm(t,dim=1), torch.linalg.norm(s, dim=1)
    t_norm_sorted, t_norm_sort_indices = torch.sort(t_norm, dim=0)
    _, s_norm_sort_indices = torch.sort(s_norm, dim=0)
    s, t = s[s_norm_sort_indices], t[t_norm_sort_indices]

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    labels = torch.zeros(t.shape, device=DEVICE)

    print("\nComputing cosine similarities.")

    for i in tqdm(range(s.shape[0])):
        s_i = s[i]

        if type == "angular_dist":
            # min angular dist
            # ---------------------
            res = torch.div(torch.arccos(cos(s_i, t)), torch.pi)
            min_ix = torch.argmin(res)
            t_closest_angular = t[min_ix]
            labels[i] = t_closest_angular
            t = torch.cat((t[0:min_ix], t[min_ix + 1:]))  # what remains

        if type == "cosine":
            # max cosine similarity
            #-----------------------
            res = cos(s_i, t)
            max_ix = torch.argmax(res)
            t_closest_cos_sim = t[max_ix]
            labels[i] = t_closest_cos_sim
            t = torch.cat((t[0:max_ix], t[max_ix + 1:]))  # what remains

    labels = labels.type(torch.float32)

    return s, labels
