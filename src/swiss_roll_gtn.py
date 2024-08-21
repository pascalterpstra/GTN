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

import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp

from torch.utils.data import DataLoader
import torch.nn as nn

name = 'swiss_roll'
dataset_path = '../data/swiss_roll/'
out_folder_base = '../out/swiss_roll/'
out_folder_train = out_folder_base + 'train/'
out_folder_val = out_folder_base + 'val/'
out_folder_test = out_folder_base + 'test/'
out_folder_weights = out_folder_base + 'weights/'

make_dirs([out_folder_train, out_folder_val, out_folder_test, out_folder_weights])

x_dim = 1
hidden_dim = x_dim + 4  # why +4? This is the minimum -- see p. 2 on RHS in https://arxiv.org/pdf/2305.18460.pdf
latent_dim = x_dim
lr = 1e-3
batch_size = 250
epochs = 500
tolerance = 3

assert os.path.exists(dataset_path + 'train.pt'), ("Did you run swiss_roll_prep.py first? Couldn't find {}"
                                                   .format(dataset_path + 'train.pt'))
train_dataset = NormalToTrgtDataset(trgt_filepath=dataset_path + 'train.pt', dataset_path=dataset_path, transform=None, subset='train')
val_dataset = NormalToTrgtDataset(trgt_filepath=dataset_path + 'val.pt', dataset_path=dataset_path, transform=None, subset='val')
test_dataset = NormalToTrgtDataset(trgt_filepath=dataset_path + 'test.pt', dataset_path=dataset_path, transform=None, subset='test')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=0, shuffle=True)

train_mean, train_std = torch.load(dataset_path+'train_mean.pt'), torch.load(dataset_path+ 'train_std.pt')


class h_hat(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(h_hat, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

        self.activation = nn.LeakyReLU(0.5)

    def forward(self, x):
        h = self.activation(self.fc1(x))
        h = self.activation(self.fc2(h))
        h = self.activation(self.fc3(h))
        x_hat = self.fc4(h)
        return x_hat


from torch.optim import Adam

def loss_function(x, x_hat):
    reproduction_loss = nn.functional.mse_loss(x_hat, x)
    return reproduction_loss

model = h_hat(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim).to(DEVICE)

optimizer = Adam(model.parameters(), lr=lr)

best_val_loss, count_val_loss_plateau = np.inf, 0


print("Starting to train GTN...")
for epoch in range(epochs):

    model.train()

    aggregate_train_loss = 0

    for batch_idx, (x, y) in enumerate(train_loader):

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        y_hat = model(x)
        loss = loss_function(y, y_hat)

        aggregate_train_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = aggregate_train_loss / (batch_idx * batch_size)
    print("Epoch {} complete. Avg train loss: {}.".format(epoch, avg_train_loss))

    y_hat = y_hat * train_std + train_mean
    y_hat = y_hat.detach().numpy().flatten()

    # validation

    model.eval()
    overall_val_loss = 0

    with torch.no_grad():

        for batch_idx, (x, y) in enumerate(val_loader):

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_hat = model(x)
            loss = loss_function(y, y_hat)

            overall_val_loss += loss.item()

        avg_val_loss = overall_val_loss / (batch_idx * batch_size)
        print("Average val loss: {}.".format(avg_val_loss))

        if avg_val_loss < best_val_loss:
            print("Improved!")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), out_folder_weights + 'weights.pt')
            count_val_loss_plateau = 0

            y_hat = y_hat * train_std + train_mean
            y_hat = y_hat.detach().numpy().flatten()

            y = y * train_std + train_mean
            y = y.detach().numpy().flatten()

            x = x.detach().numpy().flatten()

            zipped = zip(x, y_hat)
            zipped = list(zipped)

            c = mcp.gen_color(cmap="PiYG", n=len(x))

            # Using sorted and lambda
            sorted_by_x = sorted(zipped, key=lambda l: l[0])
            x, y_hat_sorted_by_x = list(zip(*sorted_by_x))

            ax = plt.axes()
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.yaxis.set_visible(False)
            plt.scatter(x, [-0.8 for i in range(len(x))], c=c)
            plt.ylim(-1, 1)
            plt.xlim(-3, 3)
            plt.savefig(out_folder_val + 'randn_epoch_{}'.format(epoch))
            plt.close()

            xs, ys = swiss_roll(theta=np.array(y_hat_sorted_by_x))
            plt.scatter(xs, ys, c=c)
            plt.ylim(-12, 15)
            plt.xlim(-12, 15)
            plt.axis('off')
            plt.savefig(out_folder_val + 'swiss_pred_epoch_{}'.format(epoch))
            plt.close()

            xs, ys = swiss_roll(theta=np.array(y))
            plt.scatter(xs, ys)
            plt.ylim(-12, 15)
            plt.xlim(-12, 15)
            plt.savefig(out_folder_val + 'swiss_gt_epoch_{}'.format(epoch))
            plt.close()

        else:
            count_val_loss_plateau += 1

        if count_val_loss_plateau > tolerance:
            break

# TESTING (generating from random using best weights and plotting)
model.eval()
model.load_state_dict(torch.load(out_folder_weights + 'weights.pt', map_location=torch.device(DEVICE)))
with torch.no_grad():

    x = torch.randn((batch_size, 1))
    y_hat = model(x)

    y_hat = y_hat * train_std + train_mean
    y_hat = y_hat.detach().numpy().flatten()

    x = x.detach().numpy().flatten()

    zipped = zip(x, y_hat)
    zipped = list(zipped)
    # print(str(zipped))

    c = mcp.gen_color(cmap="PiYG", n=len(x))

    # Using sorted and lambda
    sorted_by_x = sorted(zipped, key=lambda l: l[0])
    x, y_hat_sorted_by_x = list(zip(*sorted_by_x))

    ax = plt.axes()#[0, 0, 1, 1], frameon=False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_visible(False)
    plt.scatter(x, [-0.8 for i in range(len(x))], c=c)
    plt.ylim(-1, 1)
    plt.xlim(-3, 3)
    plt.savefig(out_folder_test + 'randn_test_epoch_{}'.format('test'))
    plt.close()

    xs, ys = swiss_roll(theta=np.array(y_hat_sorted_by_x))
    plt.scatter(xs, ys, c=c)
    plt.ylim(-12, 15)
    plt.xlim(-12, 15)
    plt.savefig(out_folder_test + 'swiss_pred_epoch_{}'.format('test'))
    plt.close()
