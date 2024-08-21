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


def make_uniform_2d_dataset(dir_dataset, n_samples):

    dir_out = dir_dataset
    make_dirs([dir_out])

    size_train, size_val, size_test = n_samples, n_samples//5, n_samples//5  # to mimic the usual 20% val, test

    train = torch.rand(size=(size_train, 2))
    val = torch.rand(size=(size_val, 2))
    test = torch.rand(size=(size_test, 2))

    save_dataset_as_torch(train, val, test, dir_out, scale_01=True)

if __name__ == '__main__':
    print("Preparing the data.")
    make_uniform_2d_dataset('../data/2d_uniform/', 100000)
    print("Done.")