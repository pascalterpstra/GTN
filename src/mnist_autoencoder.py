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

import torch.nn as nn
import torch.nn.functional as F

from utils import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.utils.data import SubsetRandomSampler

# SETTINGS

d = d_mnist
batch_size = 128
epochs = 200
tolerance = 30
lr = 1e-3
weight_decay = 1e-5

# Paths to data
data_dir_base = '../data/MNIST/'
data_dir_dim = data_dir_base + 'd_{}/'.format(d)

# Paths for saving encoded vectors and weights
weights_path = '../res/MNIST_autoencoder_weights_d_{}.pt'.format(d)
save_encoded_dir = data_dir_dim + 'MNIST_{}_images_encoded_d_{}.pt'


# DEFINING THE AUTOENCODER - encoder contains x2 2D convs with ReLU activation followed by one fully-connected layer


class Encoder(nn.Module):
    def __init__(self):

        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)  # H, W = (28+2*1-1*(4-1)-1)/2 + 1  = 14  (see formula under section "shape" here: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
        self.fc = nn.Linear(128 * 7 * 7, d)  # H, W = (14+2*1-1*(4-1)-1)/2 + 1  = 7  (see link above)

        self.activation = nn.ReLU()

    def forward(self, x):

        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(d, 128 * 7 * 7)
        self.conv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.conv1 = nn.ConvTranspose2d(64, 1, 4, 2,1)
        self.activation = nn.ReLU()
        self.activation_out = nn.Tanh()  # keep centred around 0 since data is centered around 0

    def forward(self, x):

        x = self.fc(x)
        x = x.view(x.shape[0], 128, 7, 7)
        x = self.activation(self.conv2(x))
        x = self.activation_out(self.conv1(x))
        return x


class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':

    # SETUP

    make_dirs(['../res/', data_dir_dim])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)  # center around 0
    ])

    train_dataset = MNIST(root=data_dir_base, download=True, train=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MNIST(root=data_dir_base, download=True, train=False, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # preparing to split train_dataset into train and val

    val_pct = 0.2

    n_train = len(train_dataset)
    idx = list(range(n_train))
    idx_split = int(np.floor(val_pct * n_train))

    np.random.seed(random_seed)
    np.random.shuffle(idx)

    # splitting the train data into train and val using the indices above and a torch SubsetRandomSampler

    train_idx, val_idx = idx[idx_split:], idx[:idx_split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # TRAINING

    autoencoder = Autoencoder()
    autoencoder = autoencoder.to(DEVICE)

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=weight_decay)

    autoencoder.train()

    avg_loss = 0
    val_loss_avg_min = np.inf
    n_plateau = 0

    print("Starting to train autoencoder.")

    for epoch in range(epochs):
        n_batches = 0

        for x, _ in train_dataloader:

            x = x.to(DEVICE)
            x_decoded = autoencoder(x)

            loss = F.mse_loss(x_decoded, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            n_batches += 1

        avg_loss = avg_loss / n_batches
        print("Epoch {} complete. \tavg. train loss: {}".format( epoch + 1, avg_loss))

        # VALIDATION

        autoencoder.eval()

        avg_loss_val, n_batches = 0, 0
        for x, _ in val_dataloader:
            with torch.no_grad():

                x = x.to(DEVICE)
                x_decoded = autoencoder(x)

                loss = F.mse_loss(x_decoded, x)

                avg_loss_val += loss.item()

                n_batches += 1

        avg_loss_val = avg_loss_val / n_batches
        print("avg loss val: {}".format(avg_loss_val))

        if avg_loss_val < val_loss_avg_min:
            n_plateau = 0
            val_loss_avg_min = avg_loss_val
            print("Improved! Saving.")
            torch.save(autoencoder.state_dict(), weights_path.format(d))

            autoencoder.eval()

            best_state_dict = autoencoder.state_dict()

        else:
            n_plateau += 1

        if n_plateau > tolerance:
            break

    # SAVE ENCODED IMAGES POST TRAINING

    autoencoder.eval()
    autoencoder.load_state_dict(best_state_dict)

    for data_loader, data_loader_name in [(train_dataloader, 'train'), (val_dataloader, 'val'), (test_dataloader, 'test')]:
        i = 0

        for x, _ in data_loader:
            i += 1
            with torch.no_grad():

                x = x.to(DEVICE)

                x_encoded = autoencoder.encoder(x)
                if i == 1:
                    all_encoded_images = x_encoded
                else:
                    all_encoded_images = torch.cat((all_encoded_images, x_encoded), dim=0)

        print(all_encoded_images.shape)

        # save encoded images
        torch.save(all_encoded_images, save_encoded_dir.format(data_loader_name, d))
