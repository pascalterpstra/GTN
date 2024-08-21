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

import torch

random_seed = 8745
alpha = 0.3  # for scatter plots
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set the value of d (latent dimension) for both the autoencoder and gtn:
d_mnist = 5
d_celeba = 100
d_hap = 50

