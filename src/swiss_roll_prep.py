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


def make_swiss_roll_dataset(dir_dataset, n_samples):

    make_dirs([dir_dataset])

    # train set
    a = np.random.uniform(1.5*math.pi, 4.5*math.pi, size=n_samples)
    a_train = torch.tensor(sorted(a), device=DEVICE)
    a_train = torch.unsqueeze(a_train, dim=1)

    # val set
    a = np.random.uniform(1.5*math.pi, 4.5*math.pi, size=n_samples//5)  # to mimic the usual 20% val, test
    a_val = torch.tensor(sorted(a))
    a_val = torch.unsqueeze(a_val, dim=1)

    # test set
    a = np.random.uniform(1.5*math.pi, 4.5*math.pi, size=n_samples//5)  # to mimic the usual 20% val, test
    a_test = torch.tensor(sorted(a))
    a_test = torch.unsqueeze(a_test, dim=1)

    save_dataset_as_torch(a_train, a_val, a_test, dir_dataset, scale_01=True)

    return


if __name__ == '__main__':
    print("Preparing the data.")
    make_swiss_roll_dataset('../data/swiss_roll/', n_samples=100000)
    print("Done.")
