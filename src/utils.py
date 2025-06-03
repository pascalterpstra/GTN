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
    x_y is from the dataset (see paper).  It is advisable to have the data samples x_y roughly 0-centred. Note that below,
    x and y represent input and labels respectively, for consistency with typical training notation (so x below is
    actually the y from the pair (y, x_y) and y below is actually the x_y from the pair (y, x_y).
    '''
    def __init__(self, trgt_filepath, dataset_path, subset, transform=None, target_transform=None, type="cosine", n_samples_max=10 ** 8,
                 train_mean=None, train_std=None, n_clusters=1, noise_target_by=0,
                 kmeans=None, cluster_to_mean=None, cluster_to_std=None):

        self.trgt = torch.load(trgt_filepath, map_location=torch.device(DEVICE))[:n_samples_max]
        self.cluster_to_target_norms = {}
        self.d = self.trgt.shape[-1]
        if noise_target_by > 0:
            self.trgt = torch.normal(self.trgt, noise_target_by) #0.0001

        if n_clusters > 1:
            # if 'x_{}_{}_n_clusters_{}.pt'.format(subset, n_samples_max, n_clusters) in os.listdir(dataset_path) and 'y_{}_{}_n_clusters_{}.pt'.format(subset,
            #                                                                                                   n_samples_max, n_clusters) in os.listdir(
            #         dataset_path):
            #     x_filepath = dataset_path + 'x_{}_{}_n_clusters_{}.pt'.format(subset, n_samples_max, n_clusters)
            #     y_filepath = dataset_path + 'y_{}_{}_n_clusters_{}.pt'.format(subset, n_samples_max, n_clusters)
            #     print("Found dataset with cosine sim label: {} and {}. Loading.".format(x_filepath, y_filepath))
            #     self.x = torch.load(x_filepath, map_location=torch.device(DEVICE))
            #     self.y = torch.load(y_filepath, map_location=torch.device(DEVICE))
            # else:
            clusters, kmeans = make_clusters(x=self.trgt, n_clusters=n_clusters, kmeans=kmeans)
            self.kmeans = kmeans

            (self.x, self.y, self.cluster_to_mean, self.cluster_to_std,
             self.cluster_to_rays, self.cluster_to_sampling_weight) \
                = combine_labels_from_clusters(clusters, cluster_to_mean, cluster_to_std, subset, self.trgt.shape[0])
            torch.save(self.x, dataset_path + 'x_{}_{}_n_clusters_{}.pt'.format(subset, n_samples_max, n_clusters))
            torch.save(self.y, dataset_path + 'y_{}_{}_n_clusters_{}.pt'.format(subset, n_samples_max, n_clusters))

            torch.save(self.cluster_to_mean, dataset_path + '{}_cluster_to_mean_n_clusters_{}.pt'.format(subset, n_clusters))
            torch.save(self.cluster_to_std, dataset_path + '{}_cluster_to_std_n_clusters_{}.pt'.format(subset, n_clusters))
            torch.save(self.cluster_to_rays, dataset_path + '{}_cluster_to_rays_n_clusters_{}.pt'.format(subset, n_clusters))
            torch.save(self.cluster_to_sampling_weight, dataset_path + '{}_cluster_to_sampling_weight_n_clusters_{}.pt'.format(subset, n_clusters))
            torch.save(self.cluster_to_target_norms, dataset_path + '{}_cluster_to_target_norms_n_clusters_{}.pt'.format(subset, n_clusters))

        else:
            if train_mean is not None and train_std is not None:
                self.trgt = (self.trgt - train_mean) / train_std

            self.x = torch.randn(self.trgt.shape, device=DEVICE)

            # if 'x_{}_{}.pt'.format(subset, n_samples_max) in os.listdir(dataset_path) and 'y_{}_{}.pt'.format(subset,n_samples_max) in os.listdir(dataset_path):
            #     x_filepath = dataset_path + 'x_{}_{}.pt'.format(subset, n_samples_max)
            #     y_filepath = dataset_path + 'y_{}_{}.pt'.format(subset, n_samples_max)
            #     print("Found dataset with cosine sim label: {} and {}. Loading.".format(x_filepath, y_filepath))
            #     self.x = torch.load(x_filepath, map_location=torch.device(DEVICE))
            #     self.y = torch.load(y_filepath, map_location=torch.device(DEVICE))
            # else:
            self.x, self.y = create_labels(self.x, self.trgt)
            torch.save(self.x, dataset_path + 'x_{}_{}.pt'.format(subset, n_samples_max))
            torch.save(self.y, dataset_path + 'y_{}_{}.pt'.format(subset, n_samples_max))

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

def make_clusters(x, n_clusters, kmeans):
    # from torch_kmeans import KMeans
    from fast_pytorch_kmeans import KMeans

    if kmeans is not None:
        labels = kmeans.predict(x)
    else:
        # model = KMeans(n_clusters=n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0)
        labels = kmeans.fit_predict(x)

    clusters = {}
    for i in range(n_clusters):
        sample_idx_for_cluster = torch.where(labels == i)[-1]
        samples_for_cluster = x[sample_idx_for_cluster]
        clusters[i] = samples_for_cluster
    print("Done clusters")

    # identify singleton or empty clusters and remove them
    singleton_cluster_ids = []
    for c in clusters.keys():
        if len(clusters[c]) <= 1:
            singleton_cluster_ids.append(c)
    for c in singleton_cluster_ids:
        del clusters[c]
    print("Removed {} singleton or empty clusters.".format(len(singleton_cluster_ids)))

    return clusters, kmeans


def combine_labels_from_clusters(clusters, cluster_to_mean, cluster_to_std, subset, n_data):
    target = None
    if subset == 'train':
        cluster_ids = clusters.keys()
    else:
        cluster_ids = cluster_to_mean.keys()
    cluster_to_rays, cluster_to_target_norms, cluster_to_sampling_weight = {}, {}, {}
    if subset == 'train':
        cluster_to_mean, cluster_to_std = {}, {}
    for i in cluster_ids:
        if i not in clusters.keys():
            continue
        target_cluster = torch.tensor(clusters[i], device=DEVICE)
        if subset == 'train':
            cluster_to_sampling_weight[i] = target_cluster.shape[0] / n_data
        if subset == 'train':
            target_mean_cluster, target_std_cluster = torch.mean(target_cluster, dim=0), torch.std(target_cluster, dim=0)
        else:
            target_mean_cluster, target_std_cluster = cluster_to_mean[i], cluster_to_std[i]
        # target_mean_cluster = torch.mean(target_cluster, dim=0)
        target_cluster = target_cluster - target_mean_cluster #/ target_std_cluster
        s = torch.randn(target_cluster.shape, device=DEVICE)
        source_cluster, target_cluster = create_labels(s, target_cluster)
        # remove normalization from target and do the same for source:
        target_cluster = target_cluster * target_std_cluster + target_mean_cluster
        source_cluster = source_cluster * target_std_cluster + target_mean_cluster
        # target_cluster = target_cluster + target_mean_cluster
        # source_cluster = source_cluster + target_mean_cluster
        if target is None:
            target = target_cluster
            source = source_cluster
        else:
            target = torch.cat([target, target_cluster], dim=0)
            source = torch.cat([source, source_cluster], dim=0)
        cluster_to_mean[i], cluster_to_std[i] = target_mean_cluster, target_std_cluster
    # shuffling the training data
    permuted_idx = np.random.permutation(range(len(target)))
    target = target[permuted_idx]
    source = source[permuted_idx]

    return source, target, cluster_to_mean, cluster_to_std, cluster_to_rays, cluster_to_sampling_weight  # concatenates from all clusters source_cluster and target_cluster and randomly permutes them too


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


def create_labels(s, t):
    '''
    Creates the labeled pairs (y, x_y) as described in the paper.
    :param s: tensor of shape n_samples  x  d (d = latent dimension, see paper)
                representing source distribution (Y in paper) to map to target t (next param)
    :param t: tensor of shape n_samples  x  d (d = latent dimension, see paper) representing target (X in paper)
    :return: source s with matching labels from target
    '''
    print("Creating labeled data.")

    t_norm, s_norm = torch.linalg.norm(t,dim=1), torch.linalg.norm(s, dim=1)
    # sort t and s by norm
    t_norm_sorted, t_norm_sort_indices = torch.sort(t_norm, dim=0)
    _, s_norm_sort_indices = torch.sort(s_norm, dim=0)
    s, t = s[s_norm_sort_indices], t[t_norm_sort_indices]

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    labels = torch.zeros(t.shape, device=DEVICE)

    print("\nComputing cosine similarities.")

    for i in tqdm(range(s.shape[0])):
        s_i = s[i]

        # max cosine similarity
        #-----------------------
        res = cos(s_i, t)
        max_ix = torch.argmax(res)
        # max_ix = 0
        t_closest_cos_sim = t[max_ix]
        labels[i] = t_closest_cos_sim
        t = torch.cat((t[0:max_ix], t[max_ix + 1:]))  # what remains

    labels = labels.type(torch.float32)

    return s, labels

