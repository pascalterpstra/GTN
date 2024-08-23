# Generative Topological Networks (GTNs)

The code for the [paper: Generative Topological Networks](https://arxiv.org/abs/2406.15152)

If you find the paper useful, please cite:

```
@article{levy2024generative,
  title={Generative Topological Networks},
  author={Levy-Jurgenson, Alona and Yakhini, Zohar},
  journal={arXiv preprint arXiv:2406.15152},
  year={2024}
}
```

Below are instructions on how to reproduce the examples from the paper. The instructions include the following sections:

1. Swiss-Roll
2. 2D-Uniform
3. MNIST
4. CelebA
5. HaP
6. Using pretrained weights and latent representations (CelebA and HaP)
7. Misc info

#### Setup and preliminaries

The project was developed using Python 3.7. For GPU specifications, see the appendix mentioned above.

Unless stated otherwise, the assumption is that the commands are run from the src folder.

Before following any of these guidelines below, make sure you install the requirements in requirements.txt using the following command run from the base directory (above src):

pip3 install -r requirements.txt

For your convenience, the CelebA and HaP scripts contain wandb tracking. In both, you will find a placeholder for your wandb key. The wandb install version is included in the requirements.txt. If you do not wish to use wandb, simply comment out all wandb blocks of code in the CelebA and HaP scripts, and remove it from requirements.txt.

FYI (in case you require these files for your own purposes): In all examples that contain an autoencoder, the files it generates can be found under data (for the latent representations) or res (for the model weights). The GTN weights will be saved in the out folder.


## 1. Swiss-Roll

Note that if you have installed the packages in a new virtual environment, the first time the prep script below runs, it might be slow (otherwise it takes about a second).

First run the script that prepares the data (cd to src first):
python3 swiss_roll_prep.py

Then run the script that trains the GTN:
python3 swiss_roll_gtn.py

You can navigate to the out/test folder to see the randomly generated samples all plotted together in a scatter plot using the trained GTN.

## 2. 2D-Uniform

First run the script that prepares the data:
python3 uniform_prep.py

Then run the script that trains the GTN:
python3 uniform_gtn.py

You can navigate to the out/test folder to see the randomly generated samples all plotted together in a scatter plot using the trained GTN.

## 3. MNIST

The default dimension is d=5. If you wish to change this, you can do so in conf.py.

First run the autoencoder (we recommend using a GPU for this).
python3 mnist_autoencoder.py

Then run the script that trains the GTN (doesn't require a GPU):
python3 mnist_gtn.py

You can navigate to the out/<run_name> folder to see the training process, including the randomly generated samples across epochs of improvement, as described in the paper.

## 4. CelebA

Download the file named: img_align_celeba.zip from [this link](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ) (reachable via the Google Drive link mentioned [here](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg&usp=sharing) --> Img --> img_align_celeba.zip) to the data/CelebA folder and unzip there so that your folder hierarchy looks like this:

 ```
├── data
│   ├── CelebA
│   │   ├── img_align_celeba
```


### Changing the default settings (skip to next section if not needed)
The default dimension is d=100. This can be modified in conf.py.
Other default settings can be changed either at the top of the .py files, or (in the celeba_gtn.py file) you can comment out the lines using sys arguments at the top and pass system arguments instead. Note that the setting n_hidden_layers has no effect besides controlling the name of the output folder.

### Training
In both of the following commands you can track the progress using the built-in wandb script by adding your wandb key in the designated placeholder ('YOUR_KEY') in each of these .py files.

First run the autoencoder (use a GPU for this):
python3 celeba_autoencoder.py

Then run the script that trains the GTN (use a GPU for this):
python3 celeba_gtn.py

Note that the split into data subsets is already included under data/csv, and was created using the file 'list_eval_partition.txt' that can be found under the same Google Drive link mentioned above --> Eval folder.

You can see the generated samples for every epoch of improvement in IS in the wandb run. Alternatively, you can generate samples by setting: train = False and generate_single_samples = True in celeba_gtn.py. The output can be found in out/generated.

### Generating individual samples (besides the above wandb output)

In the file celeba_gtn.py you will see settings at the top. You can set train=False. Doing so and running celeba_gtn.py will generate two folders in the out/CelebA folder -- real and generated. There you can find both real decoded images and generated images, respectively. You can also change other settings there, e.g. to produce interpolations you can set interpolate=True. 

## 5. HaP

Download the file 'archive' using the download button in this [link](https://www.kaggle.com/datasets/shyambhu/hands-and-palm-images-dataset) to the data/HaP folder and extract it there. Rename the extracted folder to 'hands_kaggle' so that your folder hierarchy looks like this:

 ```
├── data
│   ├── HaP
│   │   ├── hands_kaggle
│   │   │   ├── ...
 ```

In both of the following commands you can track the progress using the built-in wandb script by adding your wandb key in the designated placeholder ('YOUR_KEY') in each of these .py files.

### Changing the default settings (skip to next section if not needed)

The default dimension is d=50. This can be modified in conf.py.
Other default settings can be changed either at the top of the .py files, or (in the hap_gtn.py file) you can uncomment the lines using sys arguments at the top so that you can pass system arguments through the terminal instead. Note that the setting n_hidden_layers has no effect besides controlling the name of the output folder.

### Training

In both of the following commands you can track the progress using the built-in wandb script by adding your wandb key in the designated placeholder ('YOUR_KEY') in each of these .py files.

First run the autoencoder (use a GPU for this):
python3 hap_autoencoder.py

Then run the script that trains the GTN (use a GPU for this):
python3 hap_gtn.py

Note that the split into data subsets is already included under data/HaP, and was obtained using a random assignment.

You can see the generated samples for every epoch of improvement in IS in the wandb run. Alternatively, you can generate samples by setting: train = False and generate_single_samples = True in hap_gtn.py. The output can be found in out/generated.

### Generating individual samples (besides the above wandb output)

Follow the instructions described earlier for CelebA applied to hap_gtn.py instead of to celeba_gtn.py.

## 6. Using pretrained weights and latent representations:

You can skip the autoencoder training to immediately train the GTN by downloading the autoencoder weights and latent representations from the links below, placing them in the correct folders as described, and then running <DATASET_NAME>_gtn.py. 

You can also skip training the GTN to immediately start generating samples by also downloading the GTN weights (in addition to the autoencoder weights and latent representations), setting train=False in <DATASET_NAME>_gtn.py and running it. This will start to produce both real and generated samples in the ```out/<DATASET_NAME>/<RUN_NAME>``` folder.

### Autoencoder weights:

Download the autoencoder weights for the desired dataset and dimension (look for "..d_XX.." in the file name) from the following links, and place them in the  ```res``` folder:
 - [CelebA autoencoder weights](https://drive.google.com/drive/folders/1LXm5masaegMo4LnZgQVuJorButo6vwSk?usp=sharing)
 - [HaP autoencoder weights](https://drive.google.com/drive/folders/1sft-YyiQCywT8_7xMuJrCHodL8VOVthl?usp=sharing)

### Latent representations (encoded data):

Download the folder containing the encoded data for the desired dataset and dimension from the following links, and place the folder under ```data/<DATASET_NAME>``` (e.g. for CelebA, this should result in: ```data/CelebA/d_XX/```):
- [CelebA encoded data](https://drive.google.com/drive/folders/1kwPJvJ4bLXZBAUzju35bmrz4MP9HXV9Q?usp=drive_link)
- [HaP encoded data](https://drive.google.com/drive/folders/1dxjcZ8F3bSec0iS8udYPu9D_m8hn1xri?usp=sharing)

### GTN weights:

Download the folder containing the GTN weights for the desired dataset and dimension from the following links, and place the folder under ```out/<DATASET_NAME>``` (e.g. for CelebA, this should result in: ```data/CelebA/size_64_d_XX../```  and for HaP: ```data/HaP/d_XX.../```):
- [CelebA GTN weights](https://drive.google.com/drive/folders/1-lOB4wAVk2Yh6Mkm_KZJQxbATaytiWwo?usp=drive_link)
- [HaP GTN weights](https://drive.google.com/drive/folders/1rNtkb9-0XwRlxmnh3i318Jy2Bjta5yDU?usp=sharing)


## 7. Misc info

IS calculations use the included src->IScore code which was obtained from [here](https://github.com/sbarratt/inception-score-pytorch).

FID calculations were performed using [this package](https://github.com/mseitzer/pytorch-fid) applied to the paths to the folders: ```real``` and ```generated``` that are created in the relevant out subfolder (see earlier sections on generating samples). See the paper for additional info.   