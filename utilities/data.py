import torch
from torch import utils
import pandas as pd
import PIL

import pathlib
import os

class MEM_Dataset(torch.utils.data.Dataset):
    def __init__(self, features_file, img_dir, transform=None, target_transform=None):
        features_file = pd.read_csv(features_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        features_file = features_file.drop(['MMid', 'nfix', 'fixends', 'fixlengths', 'nfixes', 'npres', 'avgM', 'centerdist', 'refixdist', 'avgS'], axis=1)
        cleaned_features = features_file.groupby(['sbj', 'trial', 'image']).agg({
            'fixstarts' : list,
            'xloc' : list,
            'yloc' : list,
            'fixdur' : list,
            'scamp' : list,
            'resp' : 'first',
            'phase' : 'first'
        }).reset_index()

        self.samples = []

        len_dependent_fields = ['xloc', 'yloc', 'fixdur', 'scamp']
        window_length = 4

        for row in cleaned_features.iterrows():
            feature_dict = row[1]

            sample = {
                'sbj' : feature_dict['sbj'],
                'trial' : feature_dict['trial'],
                'image' : feature_dict['image'],
                'resp' : feature_dict['resp'],
                'phase' : feature_dict['phase'],
            }

            total_length = len(feature_dict['xloc'])

            l = 0
            r = 0

            while r < total_length:
                wl = abs(r - l) + 1
                num_padding = max(window_length - wl, 0)
                prefix = [-1] * num_padding
                for field in len_dependent_fields:
                    suffix = feature_dict[field][l:r + 1]
                    sample[field] = torch.tensor(prefix + suffix)
                
                self.samples.append(sample.copy())

                if window_length - wl > 0:
                    r += 1
                else:
                    r += 1
                    l += 1
            break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        target_sample = self.samples[idx]

        features = {k : v for k, v in target_sample.items() if k != 'phase'}
        label = target_sample['phase']

        return features, label