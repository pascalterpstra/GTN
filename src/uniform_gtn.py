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

from mycolorpy import colorlist as mcp
import random

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from utils import *

# dataset_path = '../data/2d_uniform/'
# out_folder_base = '../out/2d_uniform/'
dataset_path = '../data/2d_two_disjoint_uniforms/'
out_folder_base = '../out/2d_two_disjoint_uniforms/'

out_folder_train = out_folder_base + 'train/'
out_folder_val = out_folder_base + 'val/'
out_folder_test = out_folder_base + 'test/'
out_folder_weights = out_folder_base + 'weights/'

make_dirs([out_folder_train, out_folder_val, out_folder_test, out_folder_weights])

x_dim = 2
d = x_dim
hidden_dim = 10
latent_dim = x_dim
lr = 1e-3
batch_size = 250
epochs = 500
tolerance = 3
n_clusters = 100
noise_target_by = 0.01

assert os.path.exists(dataset_path + 'train.pt'), ("Did you run <NAME_OF_DATA>_prep.py first? Couldn't find {} "
                                                   "-- make sure you ran the correct one for the chosen dataset_path."
                                                   .format(dataset_path + 'train.pt'))
train_dataset = NormalToTrgtDataset(trgt_filepath=dataset_path + 'train.pt', dataset_path=dataset_path, transform=None,
                                    subset='train', n_clusters=n_clusters, noise_target_by=noise_target_by)
val_dataset = NormalToTrgtDataset(trgt_filepath=dataset_path + 'val.pt', dataset_path=dataset_path, transform=None,
                                  subset='val', n_clusters=n_clusters, cluster_to_mean=train_dataset.cluster_to_mean,
                                  cluster_to_std=train_dataset.cluster_to_std, noise_target_by=noise_target_by,
                                  kmeans=train_dataset.kmeans)
test_dataset = NormalToTrgtDataset(trgt_filepath=dataset_path + 'test.pt', dataset_path=dataset_path, transform=None,
                                   subset='test', n_clusters=n_clusters, cluster_to_mean=train_dataset.cluster_to_mean,
                                   cluster_to_std=train_dataset.cluster_to_std, noise_target_by=noise_target_by,
                                   kmeans=train_dataset.kmeans)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=0, shuffle=True)

print(len(train_dataset))
print(len(val_dataset))
train_mean, train_std = torch.load(dataset_path+'train_mean.pt'), torch.load(dataset_path+ 'train_std.pt')

cluster_to_sampling_weight = torch.load(dataset_path + 'train_cluster_to_sampling_weight_n_clusters_{}.pt'.format(n_clusters))
cluster_to_mean, cluster_to_std = torch.load(dataset_path + 'train_cluster_to_mean_n_clusters_{}.pt'.format(n_clusters)), torch.load(dataset_path + 'train_cluster_to_std_n_clusters_{}.pt'.format(n_clusters))


def make_global_samples_from_clusters(n_samples, cluster_to_mean, cluster_to_std ,cluster_to_sampling_weight):
    # sampling from target rays in each cluster, with number of samples from cluster depending on size of cluster
    # assert n_samples >= n_clusters, "Number of samples to generate smaller than number of clusters!"
    s_samples_global = []

    idx = [i for i in cluster_to_mean.keys() if i in cluster_to_sampling_weight.keys()]
    random.shuffle(idx)

    for id_cluster in idx:
        m = cluster_to_mean[id_cluster]
        n_to_sample_from_cluster = int(np.ceil(n_samples * cluster_to_sampling_weight[id_cluster]))

        # sampling randomly in cluster
        s_samples_local_cluster = torch.randn((n_to_sample_from_cluster, x_dim)) * cluster_to_std[id_cluster]

        # adding m to local samples, currently treated as centred around cluster mean, to obtain global samples (from
        # a vector with m being its origin to a vector with '0' being its origin)
        s_samples_global_cluster = m + s_samples_local_cluster
        s_samples_global.append(s_samples_global_cluster)

    all_samples_global = torch.cat(s_samples_global)
    randomly_chosen_ix = random.sample(range(len(all_samples_global)), n_samples)
    all_samples_global = all_samples_global[randomly_chosen_ix, :]
    return all_samples_global

class h_hat(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(h_hat, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, output_dim)

        self.activation = nn.LeakyReLU(0.5)

    def forward(self, x):
        h = self.activation(self.fc1(x))
        h = self.activation(self.fc2(h))
        h = self.activation(self.fc3(h))
        h = self.activation(self.fc4(h))
        h = self.activation(self.fc5(h))
        x_hat = self.fc6(h)
        return x_hat

model = h_hat(input_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim).to(DEVICE)


def loss_function(x, x_hat):
    reproduction_loss = F.mse_loss(x_hat, x)
    return reproduction_loss


optimizer = Adam(model.parameters(), lr=lr)

print("Starting to train GTN...")
model.train()

best_val_loss, count_val_loss_plateau = np.inf, 0

for epoch in range(epochs):
    model.train()

    overall_train_loss, overall_pval = 0, 0

    for batch_idx, (x, y) in enumerate(train_loader):

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        y_hat = model(x)
        loss = loss_function(y, y_hat)

        overall_train_loss += loss.item()

        loss.backward()
        optimizer.step()

    n_batches = batch_idx
    avg_train_loss = overall_train_loss / ((batch_idx+1) * batch_size)
    print("Epoch {} complete. Avg train loss: {}.".format(epoch, avg_train_loss))

    # PLOTTING

    y_hat = y_hat.detach().numpy()
    y = y.detach().numpy()
    x = x.detach().numpy()

    x_norm = np.array([np.linalg.norm(x[i]) for i in range(len(x))])
    zipped = zip(x, x_norm, y_hat)
    zipped = list(zipped)
    sorted_by_norm_x = sorted(zipped, key=lambda l: l[1])
    x, _, y_hat = list(zip(*sorted_by_norm_x))
    x, y_hat = np.array(x), np.array(y_hat)

    c = mcp.gen_color(cmap="Oranges", n=len(x))

    plt.scatter(x[:,0], x[:,1], c=c)
    plt.savefig(out_folder_train + 'rand_epoch_{}'.format(epoch))
    plt.close()

    xs, ys = y_hat[:, 0], y_hat[:, 1]
    plt.scatter(xs, ys, c=c)
    plt.savefig(out_folder_train + 'pred_epoch_{}'.format(epoch))
    plt.close()

    xs, ys = y[:, 0], y[:, 1]
    plt.scatter(xs, ys)
    plt.savefig(out_folder_train + 'gt_epoch_{}'.format(epoch))
    plt.close()

    # VALIDATION

    model.eval()
    overall_val_loss = 0

    with torch.no_grad():

        for batch_idx, (x, y) in enumerate(val_loader):

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_hat = model(x)
            loss = loss_function(y, y_hat)

            overall_val_loss += loss.item()

        avg_val_loss = overall_val_loss / ((batch_idx+1) * batch_size)
        print("Average val loss: {}.".format(avg_val_loss))

        if avg_val_loss < best_val_loss:
            print("Improved!")
            torch.save(model.state_dict(), out_folder_weights + 'weights_batch_{}_tolerance_{}.pt'.format(batch_size, tolerance))

            best_val_loss = avg_val_loss
            count_val_loss_plateau = 0

            # sampling from target rays
            x = make_global_samples_from_clusters(batch_size,
                                                  cluster_to_mean, cluster_to_std, cluster_to_sampling_weight)

            y_hat = model(x)

            y_hat = y_hat.detach().numpy()
            x = x.detach().numpy()

            x_norm = np.array([np.linalg.norm(x[i]) for i in range(len(x))])
            zipped = zip(x, x_norm, y_hat)
            zipped = list(zipped)
            sorted_by_norm_x = sorted(zipped, key=lambda l: l[1])
            x, _, y_hat = list(zip(*sorted_by_norm_x))
            x, y_hat = np.array(x), np.array(y_hat)

            c = mcp.gen_color(cmap="Oranges", n=len(x))

            plt.scatter(x[:, 0], x[:, 1], c=c)
            plt.savefig(out_folder_val + 'rand_epoch_{}'.format(epoch))
            plt.close()

            xs, ys = y_hat[:, 0], y_hat[:, 1]
            plt.scatter(xs, ys, c=c)
            plt.savefig(out_folder_val + 'pred_epoch_{}'.format(epoch))
            plt.close()

        else:
            count_val_loss_plateau += 1

        if count_val_loss_plateau > tolerance:
            break

# TESTING (generating using best weights and plotting)

model.eval()
model.load_state_dict(torch.load(out_folder_weights + 'weights_batch_{}_tolerance_{}.pt'.format(batch_size, tolerance), map_location=torch.device(DEVICE)))

with torch.no_grad():

    # sampling from clusters by weight
    x = make_global_samples_from_clusters(1000,
                                          cluster_to_mean, cluster_to_std, cluster_to_sampling_weight)
    y_hat = model(x)

    y_hat = y_hat.detach().numpy()

    x = x.detach().numpy()

    x_norm = np.array([np.linalg.norm(x[i]) for i in range(len(x))])
    zipped = zip(x, x_norm, y_hat)
    zipped = list(zipped)
    sorted_by_norm_x = sorted(zipped, key=lambda l: l[1])
    x, _, y_hat = list(zip(*sorted_by_norm_x))
    x, y_hat = np.array(x), np.array(y_hat)

    c = mcp.gen_color(cmap="Oranges", n=len(x))

    plt.scatter(x[:,0], x[:,1], c=c)
    plt.savefig(out_folder_test + 'rand_epoch_{}_batch_{}_tolerance_{}'.format(epoch, batch_size, tolerance))
    plt.close()

    xs, ys = y_hat[:, 0], y_hat[:, 1]
    plt.scatter(xs, ys, c=c)
    plt.savefig(out_folder_test + 'pred_epoch_{}_batch_{}_tolerance_{}'.format(epoch, batch_size, tolerance))
    plt.close()