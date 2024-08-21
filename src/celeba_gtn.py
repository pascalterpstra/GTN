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
import time
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from utils import calc_normalization_stats_from_train
from IScore.inception_score import inception_score

from celeba_autoencoder import Autoencoder


# SETTINGS

# import sys
# size, hidden_dim, lr, n_hidden_layers, train = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]).lower() == 'true'
size, hidden_dim, lr, n_hidden_layers, train = 64, 1200, 5e-5, 26, True

wandb_key = None  # TODO: if you want to use wandb, put your key string instead of None (YOUR_KEY)
if wandb_key:
    wandb.login(key=wandb_key)

d = d_celeba   # dimension of autoencoder's latent space
x_dim = d
y_dim = d
batch_size = 200
epochs = 10000
tolerance = 300
n_for_metrics = 400
inception_dims = 2048
attribute_name = None
save_weights = True

# Paths to data, out, autoencoder weights, h_hat weights
weights_autoencoder_path = '../res/CelebA_{}_autoencoder_weights_d_{}.pt'.format(size, d)
weights_h_hat_name = 'weights_d_{}'.format(d)
dataset_base_path = '../data/CelebA/'
out_base_path = '../out/'
dataset_path = dataset_base_path+'d_{}/'.format(d)
out_folder_dataset = out_base_path+'CelebA/'.format(attribute_name)
out_folder_run = (out_folder_dataset + 'size_{}_d_{}_hid_{}_lr_{}_btch_{}_epc_{}_n_hid_{}/'
                  .format(size, d, hidden_dim, lr, batch_size, epochs, n_hidden_layers))
out_folder_train = out_folder_run + 'train/'
out_folder_val = out_folder_run + 'val/'
out_folder_test = out_folder_run + 'test/'
model_weights_path = out_folder_run + weights_h_hat_name + '.pt'
if attribute_name is None:
    encoded_train_data_path = dataset_path + 'CelebA_{}_train_images_encoded_d_{}.pt'.format(size, d)
    encoded_val_data_path = dataset_path + 'CelebA_{}_val_images_encoded_d_{}.pt'.format(size, d)
    encoded_test_data_path = dataset_path + 'CelebA_{}_test_images_encoded_d_{}.pt'.format(size, d)
else:
    print("Running for attribute: {}".format(attribute_name))
    encoded_train_data_path = dataset_path + 'CelebA_{}_{}_train_images_encoded_d_{}.pt'.format(size, attribute_name, d)
    encoded_val_data_path = dataset_path + 'CelebA_{}_{}_val_images_encoded_d_{}.pt'.format(size, attribute_name, d)
    encoded_test_data_path = dataset_path + 'CelebA_{}_{}_test_images_encoded_d_{}.pt'.format(size, attribute_name, d)

# Misc evaluation options post-training. Need to set train = False (see above) for these to take effect.
generate_single_samples = True
create_single_image_real = True
interpolate, n_lambdas = False, 20
test_time = False
create_singles_for_gif = False


train_mean, train_std = calc_normalization_stats_from_train(encoded_train_data_path, dataset_path)

make_dirs([out_folder_run, out_folder_train, out_folder_val, out_folder_test])


# Defining h_hat, allowing for flexibility to choose layers (not all layers defined in the constructor
# are used in the forward function for the given setup).
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
        # self.bn21 = nn.BatchNorm1d(hidden_dim)
        self.fc22 = nn.Linear(hidden_dim, hidden_dim)
        # self.bn22 = nn.BatchNorm1d(hidden_dim)
        self.fc23 = nn.Linear(hidden_dim, hidden_dim)
        # self.bn23 = nn.BatchNorm1d(hidden_dim)
        self.fc24 = nn.Linear(hidden_dim, hidden_dim)
        # self.bn24 = nn.BatchNorm1d(hidden_dim)
        self.fc25 = nn.Linear(hidden_dim, hidden_dim)
        # self.bn25 = nn.BatchNorm1d(hidden_dim)
        self.fc26 = nn.Linear(hidden_dim, hidden_dim)
        # self.bn26 = nn.BatchNorm1d(hidden_dim)
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
        # h = self.bn21(h)
        h = self.activation(self.fc22(h))
        # h = self.bn22(h)
        h = self.activation(self.fc23(h))
        # h = self.bn23(h)
        h = self.activation(self.fc24(h))
        # h = self.bn24(h)
        h = self.activation(self.fc25(h))
        # h = self.bn25(h)
        x_hat = self.fc_out(h)
        # return x_hat
        return x_hat


model = h_hat(input_dim=d, hidden_dim=hidden_dim, output_dim=x_dim).to(DEVICE)
print("n parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))


def loss_function(y, y_hat):
    regression_loss = F.mse_loss(y, y_hat)
    return regression_loss


def generate_samples(x_all, h_hat, autoencoder,
                     weights_autoencoder_path, epoch, plot):

    autoencoder.to(DEVICE)
    h_hat.to(DEVICE)

    autoencoder.eval()
    h_hat.eval()

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
            pred_randn = h_hat(x_randn)
            pred_randn_images = autoencoder.decoder(pred_randn)
            pred_randn_images = pred_randn_images.cpu()
            imgs_randn_gen_all.append(pred_randn_images)

            if plot:
                save_images_grid(pred_randn_images[:100], out_folder_run, 'epoch_{}_pred_randn_decoded'.format(epoch))
                save_images_grid(imgs_x[:100], out_folder_run, 'epoch_{}_x'.format(epoch))

        imgs_randn_gen_all = torch.cat(imgs_randn_gen_all, dim=0)
        imgs_x_all = torch.cat(imgs_x_all, dim=0)

        return imgs_x_all, imgs_randn_gen_all


def generate_samples_time_test(n_to_gen, h_hat, autoencoder, weights_autoencoder_path):

    autoencoder.to(DEVICE)
    h_hat.to(DEVICE)

    autoencoder.eval()
    h_hat.eval()

    with (torch.no_grad()):

        # load autoencoder weights
        autoencoder.load_state_dict(torch.load(weights_autoencoder_path, map_location=torch.device(DEVICE)))

        # --------------------
        # sample normal vectors
        start = time.time()
        x_randn = torch.randn((n_to_gen, y_dim), device=DEVICE)
        pred_randn = h_hat(x_randn)
        pred_randn_images = autoencoder.decoder(pred_randn)
        end = time.time()
        print("Generating {} images took: {} seconds.".format(n_to_gen, end - start))
        return


def train_gtn():

    train_dataset = NormalToTrgtDataset(trgt_filepath=encoded_train_data_path, dataset_path=dataset_path, subset='train')
    val_dataset = NormalToTrgtDataset(trgt_filepath=encoded_val_data_path, dataset_path=dataset_path, subset='val')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                       num_workers=0, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                     num_workers=0, drop_last=True)

    model.to(DEVICE)

    optimizer = Adam(model.parameters(), lr=lr)

    print("Starting to train GTN...")
    best_val_loss, count_IS_plateau, IS_best, FD_best = np.inf, 0, 0, np.inf
    epoch_to_metrics = {}

    if wandb_key:
        run = wandb.init(
            project='CelebA GTN',
            config={
                "learning_rate": lr,
                "epochs": epochs,
                "size": size,
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

            y = y.view(batch_size, x_dim)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            y_hat = model(x)

            loss = loss_function(y, y_hat)
            overall_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = overall_train_loss / (batch_idx * batch_size)
        print("Epoch {} complete. Avg train loss: {}.".format(epoch, avg_train_loss))

        # VALIDATION
        model.eval()
        overall_val_loss, overall_val_pval = 0, 0

        with torch.no_grad():
            d = torch.load(encoded_val_data_path, map_location=torch.device(DEVICE))[:n_for_metrics]
            imgs_x, imgs_rand_gen = generate_samples(x_all=d, autoencoder=Autoencoder(),
                                                     weights_autoencoder_path=weights_autoencoder_path.format(
                                                         d),
                                                     h_hat=model, epoch=epoch, plot=False)

            print("Calculating ISs..")

            IS_gen = inception_score(imgs_rand_gen, device=DEVICE, batch_size=50, resize=True, splits=10)[0]
            print("IS gen:", IS_gen)
            IS_current = IS_gen

            if epoch == 0:
                IS_real = inception_score(imgs_x, device=DEVICE, batch_size=50, resize=True, splits=10)[0]
                print("IS real:", IS_real)

            if IS_current > IS_best:
                IS_best = IS_current
                count_IS_plateau = 0
                if save_weights:
                    print("IS improved! Saving weights and images.", IS_current)
                    torch.save(model.state_dict(), out_folder_run + 'weights_IS_epoch_{}.pt'.format(epoch))

                if wandb_key:
                    # wandb
                    imgs_IS = create_single_image_from_tensor(torchvision.utils.make_grid(imgs_rand_gen, 10, 5))
                    if epoch == 0:
                        imgs_real_0 = create_single_image_from_tensor(torchvision.utils.make_grid(imgs_x, 10, 5))
                        imgs_real_0 = wandb.Image(imgs_real_0, caption="real samples from epoch 0")
                        wandb.log({
                            "samples real 0": imgs_real_0,
                        })
                    imgs_IS = wandb.Image(imgs_IS, caption="randn sample from metric improvement")
                    wandb.log({
                               "samples IS": imgs_IS,
                               })
            else:
                count_IS_plateau += 1

            epoch_to_metrics[epoch] = [IS_real, IS_current]

            for batch_idx, (x, y) in enumerate(val_loader):

                x = x.view(batch_size, x_dim)
                x = x.to(DEVICE)

                y = y.view(batch_size, x_dim)
                y = y.to(DEVICE)

                y_hat = model(x)

                loss_val = loss_function(y, y_hat)

                overall_val_loss += loss_val.item()

            avg_val_loss = overall_val_loss / (batch_idx * batch_size)
            print("Validation complete. \tavg. val loss: {}".format(avg_val_loss))

            if avg_val_loss < best_val_loss:
                count_loss_plateau = 0
                best_val_loss = avg_val_loss

                if save_weights:
                    print("Validation loss improved! Saving weights and images.")
                    torch.save(model.state_dict(), out_folder_run + 'weights_loss_epoch_{}.pt'.format(epoch))
                if wandb_key:
                    imgs_val = create_single_image_from_tensor(torchvision.utils.make_grid(imgs_rand_gen, 10, 5))
                    imgs_val = wandb.Image(imgs_val, caption="randn sample from val improvement")
                    wandb.log({
                               "samples VAL": imgs_val,
                               })

            else:
                count_loss_plateau += 1

        if wandb_key:
            wandb.log({"train_loss": avg_train_loss,
                       "IS real":IS_real,
                       "IS gen": IS_gen,
                       "lr": lr,
                       "epochs": epochs,
                       "size": size,
                       "hidden_dim": hidden_dim,
                       "IS_best" : IS_best,
                       "Val_loss_curret": loss_val,
                       "Val_loss_best": best_val_loss,
                       "epoch":epoch+1, # note that due to how wandb logs things and how logging was used -- this will not show the correct number of epochs
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

    with torch.no_grad():
        model.load_state_dict(torch.load(model_weights_path, map_location=torch.device(DEVICE)))
        print("Model weights loaded.")
        autoencoder.load_state_dict(torch.load(weights_autoencoder_path, map_location=torch.device(DEVICE)))
        print("Autoencoder weights loaded.")

        if interpolate:
            for i in range(1000):
                start_randn, end_randn = torch.randn((1, d), device=DEVICE), torch.randn((1, d), device=DEVICE)
                print("Processing i: {}".format(i))
                ls = np.linspace(0,1,n_lambdas)

                for l in ls:
                    interpolation = l*start_randn + (1-l)*end_randn
                    interpolation_img = autoencoder.decoder(model(interpolation))
                    if l == 0:
                        all_images = interpolation_img
                    else:
                        all_images = torch.cat([all_images, interpolation_img], dim=0)

                    if create_singles_for_gif:
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

            make_dirs([out_folder_run +'generated/{}/'.format(weights_h_hat_name), out_folder_run + 'real/'])

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
                    plt.savefig(out_folder_run +'generated/{}/gen_{}.jpeg'.format(weights_h_hat_name, batch_idx * batch_size + i), bbox_inches='tight', dpi=64)
                    plt.close()

        if test_time:
            generate_samples_time_test(1000, autoencoder=Autoencoder(),
                      weights_autoencoder_path=weights_autoencoder_path.format(d), h_hat=model)