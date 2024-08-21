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

from utils import *

import numpy as np
from mycolorpy import colorlist as mcp
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F

from mnist_autoencoder import Autoencoder

# SETUP

d = d_mnist
y_dim, x_dim = d, d
hidden_dim = 50
lr = 1e-3
batch_size = 128
epochs = 200
tolerance = 3

dataset_path = '../data/MNIST/d_{}/'.format(d)
out_folder_dataset = '../out/MNIST/'
out_folder_run = (out_folder_dataset + 'd_{}_h_{}_lr_{}_btch_{}_epc_{}/'
                  .format(d, hidden_dim, lr, batch_size, epochs))
out_folder_train = out_folder_run + 'train/'
out_folder_val = out_folder_run + 'val/'
out_folder_test = out_folder_run + 'test/'

encoded_train_data_path = dataset_path + 'MNIST_train_images_encoded_d_{}.pt'.format(d)
encoded_val_data_path = dataset_path + 'MNIST_val_images_encoded_d_{}.pt'.format(d)

weights_autoencoder_path = '../res/MNIST_autoencoder_weights_d_{}.pt'.format(d)

train_mean, train_std = calc_normalization_stats_from_train(encoded_train_data_path, dataset_path)

make_dirs([out_folder_run, out_folder_train, out_folder_val, out_folder_test])


# PREPARE DATASET FOR TRAINING

# prepare Dataset object -- this is where Gaussian is chosen as Y (see paper).
train_dataset = NormalToTrgtDataset(trgt_filepath=encoded_train_data_path, dataset_path=dataset_path, subset='train')
val_dataset = NormalToTrgtDataset(trgt_filepath=encoded_val_data_path, dataset_path=dataset_path, subset='val')

train_loader_encoder = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   num_workers=0, shuffle=True, drop_last=True)
val_loader_encoder = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                 num_workers=0, drop_last=True)


# DEFINING h_hat


class h_hat(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(h_hat, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.bn5 = nn.BatchNorm1d(hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.bn6 = nn.BatchNorm1d(hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, output_dim)

        self.activation = nn.LeakyReLU(0.5)

    def forward(self, x):
        h = self.activation(self.fc1(x))
        h = self.bn1(h)
        h = self.activation(self.fc2(h))
        h = self.bn2(h)
        h = self.activation(self.fc3(h))
        h = self.bn3(h)
        h = self.activation(self.fc4(h))
        h = self.bn4(h)
        h = self.activation(self.fc5(h))
        h = self.bn5(h)
        h = self.activation(self.fc6(h))
        h = self.bn6(h)
        x_hat = self.fc7(h)
        return x_hat


model = h_hat(input_dim=d, hidden_dim=hidden_dim, output_dim=x_dim)

print("n parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))


def loss_function(y, y_hat):
    regression_loss = F.mse_loss(y, y_hat)
    return regression_loss


def generate_samples(x_all, h_hat, autoencoder, weights_autoencoder_path, epoch, plot):

    autoencoder.to(DEVICE)
    h_hat.to(DEVICE)

    autoencoder.eval()
    h_hat.eval()

    with torch.no_grad():

        # load autoencoder weights
        autoencoder.load_state_dict(torch.load(weights_autoencoder_path, map_location=torch.device(DEVICE)))

        imgs_x_all, imgs_randn_gen_all = [], []
        for i in range(0, x_all.shape[0], batch_size):
            x = x_all[i:i+batch_size]

            # reconstruct original image (from original latent vector x)
            x.to(DEVICE)
            imgs_x = autoencoder.decoder(x)
            imgs_x = imgs_x.cpu()
            imgs_x_all.append(imgs_x)

            # generate samples
            # --------------------
            # sample normal vectors y
            x_randn = torch.randn(x.shape, device=DEVICE)
            # pass them through h_hat to get predictions x_y
            pred_randn = h_hat(x_randn)
            # decode predictions
            pred_randn_images = autoencoder.decoder(pred_randn)

            # collecting resulting images and plotting some of them
            pred_randn_images = pred_randn_images.cpu()
            imgs_randn_gen_all.append(pred_randn_images)
            if plot:
                # plot both generated and real samples
                save_images_grid(pred_randn_images[:100], out_folder_run, 'epoch_{}_h_hat_randn_decoded'.format(epoch))
                save_images_grid(imgs_x[:100], out_folder_run, 'epoch_{}_x_decoded'.format(epoch))

        imgs_randn_gen_all = torch.cat(imgs_randn_gen_all, dim=0)
        imgs_x_all = torch.cat(imgs_x_all, dim=0)

        return imgs_x_all, imgs_randn_gen_all


model.to(DEVICE)
optimizer = Adam(model.parameters(), lr=lr)

print("Starting to train GTN.")
best_val_loss = np.inf

for epoch in range(epochs):

    model.train()

    overall_loss_encoder, overall_pval = 0, 0

    for batch_idx, (x, y) in enumerate(train_loader_encoder):

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        y_hat = model(x)

        loss = loss_function(y, y_hat)
        overall_loss_encoder += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = overall_loss_encoder / (batch_idx * batch_size)

    print("Epoch {} complete. \tavg. train loss: {}".format(epoch+1, avg_train_loss))

    # VALIDATION

    model.eval()
    overall_val_loss, overall_val_pval = 0, 0

    with torch.no_grad():

        for batch_idx, (x, y) in enumerate(val_loader_encoder):

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_hat = model(x)

            loss_val_encoder = loss_function(y, y_hat)

            overall_val_loss += loss_val_encoder.item()

        avg_val_loss_encoder = overall_val_loss / (batch_idx * batch_size)
        print("Validation complete. \tavg. val loss: {}".format(avg_val_loss_encoder))

        # check if improved on validation -- if so, save weights, and plot both real decoded and generated
        if avg_val_loss_encoder < best_val_loss:
            count_loss_plateau = 0
            best_val_loss = avg_val_loss_encoder

            # saving weights
            print("Validation loss improved! Saving weights and images.")
            torch.save(model.state_dict(), out_folder_run + 'weights.pt'.format(epoch))

            # getting some latent representations of real images for plotting against generated
            d = torch.load(encoded_val_data_path, map_location=torch.device(DEVICE))[:batch_size]

            # generate samples and plot both them and the above real images
            generate_samples(x_all=d, autoencoder=Autoencoder(),
                             weights_autoencoder_path=weights_autoencoder_path.format(d),
                             h_hat=model, epoch=epoch, plot=True)
        else:
            count_loss_plateau += 1

    if count_loss_plateau > tolerance:
        print("loss hasn't improved for {} rounds. Stopping.".format(tolerance))
        break
