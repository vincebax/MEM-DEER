import torch
import torchvision
import pandas as pd

IMG_HEIGHT = 768
IMG_WIDTH = 1024

class MEM_Dataset(torch.utils.data.Dataset):

    def __init__(self, features_file, img_dir, transform=None, target_scanpath_length=4): # TODO tweak value if needed

        self.phase_enc_map = {
            'aesthetic' : 0,
            'memorize' : 1
        }

        features_file = pd.read_csv(features_file)
        self.img_dir = img_dir
        self.transform = transform # TODO implement image transform support

        test_features = features_file[features_file['phase'] == 'test']
        zs_norm_map = {
            'fd_mean' : test_features['fixdur'].mean(),
            'fd_std' : test_features['fixdur'].std(),
            'scamp_mean' : test_features['scamp'].mean(),
            'scamp_std' : test_features['scamp'].std(),
        }

        cleaned_features = features_file.groupby(['sbj', 'image', 'phase']).agg({
            'xloc' : list,
            'yloc' : list,
            'fixdur' : list,
            'scamp' : list,
        }).reset_index()

        test_features = cleaned_features[cleaned_features['phase'] == 'test']
        original_features = cleaned_features[cleaned_features['phase'] != 'test']

        self.images = []
        self.scanpath_tensors = []
        self.original_tasks = []

        scanpath_length = target_scanpath_length
        scanpath_fields = ['xloc', 'yloc', 'fixdur', 'scamp']

        original_task_map = {}

        for row in original_features.iterrows():

            feature_dict = row[1]
            key = (feature_dict['sbj'], feature_dict['image'])
            if key in original_task_map:
                print('Duplicate key found')
            original_task_map[key] = feature_dict['phase']

        for row in test_features.iterrows():

            feature_dict = row[1]

            original_task = original_task_map.get((feature_dict['sbj'], feature_dict['image']), -1)
            is_valid_data = original_task != -1

            if not is_valid_data:
                continue

            img_path = img_dir / feature_dict['image']
            self.images.append(img_path)
            self.original_tasks.append(self.phase_enc_map[original_task])

            feature_tensors = []
            for field in scanpath_fields:
                feature_tensor = torch.tensor(feature_dict[field], dtype=torch.float32)

                if field == 'xloc': # normalize spatial fields
                    feature_tensor /= IMG_WIDTH
                elif field == 'yloc':
                    feature_tensor /= IMG_HEIGHT
                elif field == 'fixdur':
                    feature_tensor = (feature_tensor - zs_norm_map['fd_mean']) / zs_norm_map['fd_std']
                elif field == 'scamp':
                    feature_tensor = (feature_tensor - zs_norm_map['scamp_mean']) / zs_norm_map['scamp_std']
                    

                if feature_tensor.shape[0] > scanpath_length:
                    feature_tensor = feature_tensor[:scanpath_length]
                else:
                    pad_length = scanpath_length - len(feature_dict[field])
                    padding = torch.zeros(pad_length)
                    feature_tensor = torch.cat([feature_tensor, padding], dim=0) # TODO change handling of variable length, simple solution for now

                feature_tensors.append(feature_tensor)

            sp_tensor = torch.stack(feature_tensors, dim=1)
            self.scanpath_tensors.append(sp_tensor)

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        image = torchvision.io.read_image(self.images[idx]).float() / 255.0
        if self.transform:
            image = self.transform(image)
        # image = str(self.images[idx])
        scanpath = self.scanpath_tensors[idx]
        label = self.original_tasks[idx]

        return image, scanpath, label
    

# testing code

# import os
# from pathlib import Path
# from torchvision.transforms import v2
# transform = v2.Compose([
#     v2.Resize(size=(224,224)),
#     v2.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])
# base_path = Path(os.getcwd()) # must be in ~/DEER/ to run, TODO fix to run regardless, probably a much later fix
# img_dir = base_path / 'datasets' / 'MEM' / 'stimuli'
# features_file = base_path / 'datasets' / 'MEM' / 'features.csv'

# ds = MEM_Dataset(features_file, img_dir, transform)

# dl = torch.utils.data.DataLoader(ds, batch_size=1)
# i = 0
# for image, scanpath, label in dl:

#     print()
#     # print(image)
#     print(scanpath)
#     print(scanpath.shape)
#     print(label)
#     print()

#     if i > 5:
#         break
#     else:
#         i += 1