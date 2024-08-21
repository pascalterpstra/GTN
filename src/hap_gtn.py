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
import wandb
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from IScore.inception_score import inception_score

from hap_autoencoder import Autoencoder


wandb.login(key='YOUR_KEY')

# import sys
# hidden_dim, lr, n_hidden_layers, train = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), str(sys.argv[4]).lower() == 'true'
hidden_dim, lr, n_hidden_layers, train = 3000, 5e-5, 26, True

d = d_hap
x_dim, y_dim = d, d
batch_size = 200
epochs = 10000
tolerance = 300
n_for_metrics = 200
inception_dims = 2048

# Paths to data, out, autoencoder weights, h_hat weights
weights_autoencoder_path = '../res/hands_autoencoder_weights_d_{}.pt'.format(d)
model_weights_name = 'weights_d_{}'.format(d)
dataset_path = '../data/HaP/d_{}/'.format(d)
out_folder_dataset = '../out/HaP/'
out_folder_run = (out_folder_dataset + 'd_{}_hid_{}_lr_{}_btch_{}_epc_{}_n_hid_{}/'
                  .format(d, hidden_dim, lr, batch_size, epochs, n_hidden_layers))
out_folder_train = out_folder_run + 'train/'
out_folder_val = out_folder_run + 'val/'
out_folder_test = out_folder_run + 'test/'
encoded_train_data_path = dataset_path + 'hands_train_images_encoded_d_{}.pt'.format(d)
encoded_val_data_path = dataset_path + 'hands_val_images_encoded_d_{}.pt'.format(d)
encoded_test_data_path = dataset_path + 'hands_test_images_encoded_d_{}.pt'.format(d)
model_weights_path = out_folder_run + model_weights_name + '.pt'

# Misc evaluation options post-training. Need to set train = False (see above) for these to take effect.
generate_single_samples = True
create_single_image_real = True
create_single_real_samples_and_latent_vectors = False
interpolate, n_lambdas, use_real_start_end = False, 20, False
create_singles_for_gif = False

train_mean, train_std = calc_normalization_stats_from_train(encoded_train_data_path, dataset_path)

make_dirs([out_folder_run, out_folder_train, out_folder_val, out_folder_test])


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
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.bn7 = nn.BatchNorm1d(hidden_dim)
        self.fc8 = nn.Linear(hidden_dim, hidden_dim)
        self.bn8 = nn.BatchNorm1d(hidden_dim)
        self.fc9 = nn.Linear(hidden_dim, hidden_dim)
        self.bn9 = nn.BatchNorm1d(hidden_dim)
        self.fc10 = nn.Linear(hidden_dim, hidden_dim)
        self.bn10 = nn.BatchNorm1d(hidden_dim)
        self.fc11 = nn.Linear(hidden_dim, hidden_dim)
        self.bn11 = nn.BatchNorm1d(hidden_dim)
        self.fc12 = nn.Linear(hidden_dim, hidden_dim)
        self.bn12 = nn.BatchNorm1d(hidden_dim)
        self.fc13 = nn.Linear(hidden_dim, hidden_dim)
        self.bn13 = nn.BatchNorm1d(hidden_dim)
        self.fc14 = nn.Linear(hidden_dim, hidden_dim)
        self.bn14 = nn.BatchNorm1d(hidden_dim)
        self.fc15 = nn.Linear(hidden_dim, hidden_dim)
        self.bn15 = nn.BatchNorm1d(hidden_dim)
        self.fc16 = nn.Linear(hidden_dim, hidden_dim)
        self.bn16 = nn.BatchNorm1d(hidden_dim)
        self.fc17 = nn.Linear(hidden_dim, hidden_dim)
        self.bn17 = nn.BatchNorm1d(hidden_dim)
        self.fc18 = nn.Linear(hidden_dim, hidden_dim)
        self.bn18 = nn.BatchNorm1d(hidden_dim)
        self.fc19 = nn.Linear(hidden_dim, hidden_dim)
        self.bn19 = nn.BatchNorm1d(hidden_dim)
        self.fc20 = nn.Linear(hidden_dim, hidden_dim)
        self.bn20 = nn.BatchNorm1d(hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, hidden_dim)
        self.fc22 = nn.Linear(hidden_dim, hidden_dim)
        self.fc23 = nn.Linear(hidden_dim, hidden_dim)
        self.fc24 = nn.Linear(hidden_dim, hidden_dim)
        self.fc25 = nn.Linear(hidden_dim, hidden_dim)
        self.fc26 = nn.Linear(hidden_dim, hidden_dim)
        self.fc27 = nn.Linear(hidden_dim, hidden_dim)
        self.fc28 = nn.Linear(hidden_dim, hidden_dim)
        self.fc29 = nn.Linear(hidden_dim, hidden_dim)
        self.fc30 = nn.Linear(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, hidden_dim)
        self.fc32 = nn.Linear(hidden_dim, hidden_dim)
        self.fc33 = nn.Linear(hidden_dim, hidden_dim)
        self.fc34 = nn.Linear(hidden_dim, hidden_dim)
        self.fc35 = nn.Linear(hidden_dim, hidden_dim)
        self.fc36 = nn.Linear(hidden_dim, hidden_dim)
        self.fc37 = nn.Linear(hidden_dim, hidden_dim)
        self.fc38 = nn.Linear(hidden_dim, hidden_dim)
        self.fc39 = nn.Linear(hidden_dim, hidden_dim)
        self.fc40 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, output_dim)

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
        h = self.activation(self.fc7(h))
        h = self.bn7(h)
        h = self.activation(self.fc8(h))
        h = self.bn8(h)
        h = self.activation(self.fc9(h))
        h = self.bn9(h)
        h = self.activation(self.fc10(h))
        h = self.bn10(h)
        h = self.activation(self.fc11(h))
        h = self.bn11(h)
        h = self.activation(self.fc12(h))
        h = self.bn12(h)
        h = self.activation(self.fc13(h))
        h = self.bn13(h)
        h = self.activation(self.fc14(h))
        h = self.bn14(h)
        h = self.activation(self.fc15(h))
        h = self.bn15(h)
        h = self.activation(self.fc16(h))
        h = self.bn16(h)
        h = self.activation(self.fc17(h))
        h = self.bn17(h)
        h = self.activation(self.fc18(h))
        h = self.bn18(h)
        h = self.activation(self.fc19(h))
        h = self.bn19(h)
        h = self.activation(self.fc20(h))
        h = self.bn20(h)
        h = self.activation(self.fc21(h))
        h = self.activation(self.fc22(h))
        h = self.activation(self.fc23(h))
        h = self.activation(self.fc24(h))
        h = self.activation(self.fc25(h))

        x_hat = self.fc_out(h)
        return x_hat

model = h_hat(input_dim=d, hidden_dim=hidden_dim, output_dim=x_dim).to(DEVICE)

print("n parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))


def loss_function(y, y_hat):
    regression_loss = F.mse_loss(y, y_hat)
    return regression_loss


def generate_samples(x_all, homeo, autoencoder,
                     weights_autoencoder_path, epoch, plot):

    autoencoder.to(DEVICE)
    homeo.to(DEVICE)

    autoencoder.eval()
    homeo.eval()

    with (torch.no_grad()):

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
            # sample normal vectors
            x_randn = torch.randn(x.shape, device=DEVICE)
            pred_randn = homeo(x_randn)
            pred_randn_images = autoencoder.decoder(pred_randn)
            pred_randn_images = pred_randn_images.cpu()
            imgs_randn_gen_all.append(pred_randn_images)

            if plot:
                save_images_grid(pred_randn_images[:100], out_folder_run, 'epoch_{}_pred_randn_decoded'.format(epoch))
                save_images_grid(imgs_x[:100], out_folder_run, 'epoch_{}_x'.format(epoch))

        imgs_randn_gen_all = torch.cat(imgs_randn_gen_all, dim=0)
        imgs_x_all = torch.cat(imgs_x_all, dim=0)

        return imgs_x_all, imgs_randn_gen_all


def train_gtn():

    # # After observing that IS was a better stopping criteria than the validation set in CelebA, the validation set
    # # is no longer used, and instead images are generated and evaluated using their IS. See paper for details.
    train_dataset = NormalToTrgtDataset(trgt_filepath=encoded_train_data_path, dataset_path=dataset_path, subset='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                       num_workers=0, shuffle=True, drop_last=True)


    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr)

    print("Start training...")
    best_val_loss, count_IS_plateau, IS_best = np.inf, 0, 0
    epoch_to_metrics = {}

    run = wandb.init(
        project='HaP GTN',
        config={
            "learning_rate": lr,
            "epochs": epochs,
            "hidden_dim" : hidden_dim,
            "n_layers" : n_hidden_layers,
        },
    )

    for epoch in range(epochs):

        model.train()

        overall_train_loss, overall_pval = 0, 0

        for batch_idx, (x, y) in enumerate(train_loader):

            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)

            y = y.view(batch_size, y_dim)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            y_hat = model(x)

            loss = loss_function(y, y_hat)
            overall_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = overall_train_loss / (batch_idx * batch_size)
        print("Epoch {} complete. Avg train loss: {}.".format(epoch, avg_train_loss))

        # VALIDATION -- using IS

        model.eval()

        with torch.no_grad():
            d = torch.load(encoded_val_data_path, map_location=torch.device(DEVICE))[:n_for_metrics]
            imgs_x, imgs_rand_gen = generate_samples(x_all=d, autoencoder=Autoencoder(),
                                                     weights_autoencoder_path=weights_autoencoder_path.format(
                                                         d),
                                                     homeo=model, epoch=epoch, plot=True)

            if epoch % 5 == 0:
                imgs_train = create_single_image_from_tensor(torchvision.utils.make_grid(imgs_rand_gen, 10, 5))
                imgs_train = wandb.Image(imgs_train, caption="randn sample from epoch: {}".format(epoch))
                wandb.log({
                           "samples epoch": imgs_train,
                           })

            print("Calculating ISs..")
            imgs_rand_gen_for_IS = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(imgs_rand_gen)
            IS_gen = inception_score(imgs_rand_gen_for_IS, device=DEVICE, batch_size=50, resize=True, splits=10)[0]
            print("IS gen:", IS_gen)
            IS_current = IS_gen

            imgs_x_for_IS = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(imgs_x)
            if epoch == 0:
                IS_real = inception_score(imgs_x_for_IS, device=DEVICE, batch_size=50, resize=True, splits=10)[0]
                print("IS real:", IS_real)

            if IS_current > IS_best:
                IS_best = IS_current
                count_IS_plateau = 0
                print("IS improved! Saving weights and images.", IS_current)
                torch.save(model.state_dict(), out_folder_run + 'weights_IS_epoch_{}.pt'.format(epoch))
                imgs_IS = create_single_image_from_tensor(torchvision.utils.make_grid(imgs_rand_gen, 10, 5))
                imgs_IS = wandb.Image(imgs_IS, caption="randn sample from metric improvement")
                wandb.log({
                           "samples IS": imgs_IS,
                           })
            else:
                count_IS_plateau += 1

            epoch_to_metrics[epoch] = [IS_real, IS_current]

        wandb.log({"train_loss": avg_train_loss,
                   "IS real":IS_real,
                   "IS gen": IS_gen,
                   "learning_rate": lr,
                   "epochs": epochs,
                   "hidden_dim": hidden_dim,
                   "IS_best" : IS_best,
                   })

        if count_IS_plateau > tolerance:
            print("IS hasn't improved for {} rounds. Stopping.".format(tolerance))
            df = pd.DataFrame(epoch_to_metrics)
            df.to_csv(out_folder_run + 'epoch_to_IS.csv', index=False)
            return


if train:
    train_gtn()
else:
    print("Evaluating.")

    autoencoder = Autoencoder()
    model.to(DEVICE)
    autoencoder.to(DEVICE)

    model.eval()
    autoencoder.eval()

    with ((torch.no_grad())):
        model.load_state_dict(torch.load(model_weights_path, map_location=torch.device(DEVICE)))
        print("Model weights loaded.")
        autoencoder.load_state_dict(torch.load(weights_autoencoder_path, map_location=torch.device(DEVICE)))
        print("Autoencoder weights loaded.")

        if create_single_real_samples_and_latent_vectors:
            # normal vectors
            x_train = torch.load('../data/HaP/hands_latent_dim_50/x_train.pt', map_location=DEVICE)
            # corresponding labels for normal vectors
            y_train = torch.load('../data/HaP/hands_latent_dim_50/y_train.pt', map_location=DEVICE)

            for i in range(len(x_train)):
                v = x_train[i]
                v = torch.reshape(v, (1, v.shape[0]))
                img = autoencoder.decoder(model(v))
                folder = out_folder_run + 'real_and_vectors/'
                make_dirs([folder])
                img = create_single_image_from_tensor(torchvision.utils.make_grid(img, 10, 5))
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
                plt.imshow(img)
                plt.savefig(folder + 'real_{}.jpeg'.format(i), dpi=64, bbox_inches='tight')
                plt.close()
                torch.save(v, folder+'real_v_{}.pt'.format(i))

        if interpolate:

            start_end_pairs = [[27, 8]] # TODO: select which image IDs from the above output of single images you wish to include (e.g. if names are 27.png and 8.png then put: [[27, 8]]
            for i in range(len(start_end_pairs)):

                real_start_id, real_end_id = start_end_pairs[i][0], start_end_pairs[i][1]

                # loads the vectors for the chosen start and end images
                start_real = torch.load(out_folder_run + 'real_and_vectors/real_v_{}.pt'.format(real_start_id), map_location=DEVICE)
                end_real = torch.load(out_folder_run + 'real_and_vectors/real_v_{}.pt'.format(real_end_id), map_location=DEVICE)

                start_vec, end_vec = start_real, end_real
                print("Processing i: {}".format(i))
                ls = np.linspace(0, 1, n_lambdas)

                for l in ls:
                    interpolation = l * start_vec + (1 - l) * end_vec
                    interpolation_img = autoencoder.decoder(model(interpolation))
                    if l == 0:
                        all_images = interpolation_img
                    else:
                        all_images = torch.cat([all_images, interpolation_img], dim=0)

                    if create_singles_for_gif:
                        if use_real_start_end:
                            folder = out_folder_run + 'interpolation_real_start_end_gif_{}/'.format(i)
                        else:
                            folder = out_folder_run + 'interpolation_gif_{}/'.format(i)
                        make_dirs([folder])
                        img = create_single_image_from_tensor(torchvision.utils.make_grid(interpolation_img, 10, 5))
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
                        plt.imshow(img)

                        plt.savefig(folder + 'interpol_{}.jpeg'.format(l), dpi=64, bbox_inches='tight')
                        plt.close()

                all_images = all_images.cpu()

        if generate_single_samples:

            make_dirs([out_folder_run+'generated/{}/'.format(model_weights_name), out_folder_run+'real/'])

            train_dataset_encoded = EncodedImagesDataset(trgt_filepath=encoded_train_data_path)
            train_loader_encoded = torch.utils.data.DataLoader(train_dataset_encoded, batch_size=batch_size,
                                                               num_workers=0, shuffle=False, drop_last=True)

            for batch_idx, x in enumerate(train_loader_encoded):
                x = x.to(DEVICE)
                imgs_x = autoencoder.decoder(x)
                imgs_x = imgs_x.cpu()

                # create samples
                # --------------------
                # sample normal vectors
                x_randn = torch.randn(x.shape, device=DEVICE)
                pred_randn = model(x_randn)
                pred_randn_images = autoencoder.decoder(pred_randn)
                pred_randn_images = pred_randn_images.cpu()

                if create_single_image_real:
                    for i in range(len(imgs_x)):
                        img = create_single_image_from_tensor(torchvision.utils.make_grid(imgs_x[i], 10, 5))
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
                        plt.imshow(img)
                        plt.savefig(out_folder_run+'real/real_{}.jpeg'.format(batch_idx*batch_size+i), dpi=64)
                        plt.close()

                all_images = []
                for i in range(len(pred_randn_images)):
                    img_gen = create_single_image_from_tensor(torchvision.utils.make_grid(pred_randn_images[i], 10, 5))
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
                    plt.imshow(img_gen)
                    plt.axis('off')
                    plt.savefig(out_folder_run +'generated/{}/gen_{}.jpeg'.format(model_weights_name, batch_idx * batch_size + i), bbox_inches='tight', dpi=64)
                    plt.close()
