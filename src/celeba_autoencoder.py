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
from torchvision.utils import make_grid


# SETTINGS

batch_size = 200
d = d_celeba
epochs = 50
lr = 1e-4
weight_decay = 1e-5
tolerance = 5

# Paths to data
data_dir_base = '../data/CelebA/'
data_dir_dim = data_dir_base + 'd_{}/'.format(d)
img_dir = data_dir_base + 'img_align_celeba'
csv_dir = data_dir_base +'csv/'

# Paths for saving encoded vectors and weights
weights_path = '../res/CelebA_64_autoencoder_weights_d_{}.pt'.format(d)
save_encoded_dir = '../data/CelebA/d_{}/'.format(d)
save_encoded_path_no_attr = save_encoded_dir + 'CelebA_64_{}_images_encoded_d_{}.pt'
save_encoded_path_with_attr = save_encoded_dir + 'CelebA_64_{}_{}_images_encoded_d_{}.pt'

wandb_project_name = 'CelebA Autoencoder'


# DEFINING THE AUTOENCODER - encoder contains x2 2D convs with ReLU activation followed by one fully-connected layer

class Encoder(nn.Module):
    def __init__(self):

        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)  # H, W = (64+2*1-1*(4-1)-1)/2 + 1  = 32  (see formula under section "shape" here: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
        self.fc = nn.Linear(128 * 16 * 16, d)  # H, W = (32+2*1-1*(4-1)-1)/2 + 1  = 16  (see link above)

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

        self.fc = nn.Linear(d, 128 * 16 * 16)
        self.conv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.conv1 = nn.ConvTranspose2d(64, 3, 4, 2,1)
        self.activation = nn.ReLU()
        self.activation_out = nn.Tanh()  # keep centred around 0 since data is centered around 0

    def forward(self, x):

        x = self.fc(x)
        x = x.view(x.shape[0], 128, 16, 16)
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

    import wandb

    # TODO: insert your wandb key here. Alternatively, comment out wandb blocks of code
    wandb.login(key='YOUR_KEY')

    make_dirs([save_encoded_dir])

    custom_transform = transforms.Compose([transforms.CenterCrop((148, 148)),
                                           transforms.Resize((64, 64)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(0.5, 0.5)])

    # The csvs used below were produced from the CelebA file (from the CelebA Drive account): list_eval_partition.txt
    train_dataset = CelebaDataset(csv_path=csv_dir + 'celeba-train.csv',
                                  img_dir=img_dir,
                                  attribute_name=None,
                                  transform=custom_transform)

    val_dataset = CelebaDataset(csv_path=csv_dir + 'celeba-val.csv',
                                  img_dir=img_dir,
                                  attribute_name=None,
                                  transform=custom_transform)

    test_dataset = CelebaDataset(csv_path=csv_dir + 'celeba-test.csv',
                                 img_dir=img_dir,
                                 attribute_name=None,
                                 transform=custom_transform)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=4)

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=4)


    # TRAINING

    run = wandb.init(
        project=wandb_project_name,
        config={
            "lr": lr,
            "d": d,
            "epochs": epochs,
        },
    )

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
        print("Epoch {} complete. \tavg. train loss: {}".format(epoch + 1, avg_loss))

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

        imgs = x_decoded.cpu()
        img_grid = create_single_image_from_tensor(make_grid(imgs, 10, 5))
        img_grid = wandb.Image(img_grid, caption="val decoded")

        wandb.log({
            "val decoded": img_grid,
            "val loss": avg_loss_val,
            "train loss": avg_loss
        })

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

    for data_loader, data_loader_name in [(train_dataloader, 'train'), (val_dataloader, 'val'),
                                          (test_dataloader, 'test')]:
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

        torch.save(all_encoded_images,
                   save_encoded_path_no_attr.format(data_loader_name, d))

