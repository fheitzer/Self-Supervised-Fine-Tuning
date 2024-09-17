import torch
import PIL.Image
import os
import numpy as np
from skimage import io
from skimage.transform import resize
import random
from torchvision import transforms, datasets
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader
import pandas as pd


# Stratified Kfold to split the data balanced by class and patient for train and val

from utils import recommend_num_workers, recommend_max_batch_size

from typing import Optional

def slice_by_percentage(x, percentage, train=True):
    split = int(len(x) * percentage)
    if train:
        return x[:split]
    else:
        return x[split:]


class CustomImgFolderDataset(datasets.ImageFolder):

    def __init__(self, root: str, transform: transforms.Compose, target_transform: transforms.Compose):
        super().__init__(root=root, transform=transform, target_transform=target_transform)

    def __getitem__(self, idx):        
        path, target = self.samples[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path

class CustomMetaDataset(Dataset):
    """This dataset samples an unbalanced binary classification dataset in a 50/50 fashion"""

    def __init__(self,
                 meta_path,
                 img_dir,
                 transform: transforms.Compose,
                 target_transform: transforms.Compose,
                 attribution: str=None):
        super().__init__()
        self.df = pd.read_csv(meta_path, low_memory=False)
        if attribution:
            self.df = self.df[self.df['attribution'] == attribution]
        self.targets = self.df['target'].values
        self.image_ids = self.df['isic_id'].values
            
        self.transform = transform
        self.target_transform = target_transform
        self.img_dir = img_dir
        self.train = train
        self.n_samples = len(self.df)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Get Image and target
        img_id = self.image_ids[idx]
        img = default_loader(os.path.join(self.img_dir, img_id + '.jpg'))
        target = targets[idx]
        # Transform if configured
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
            
        return img, target, img_id
        

class CustomMetaDatasetBalanced(Dataset):
    """This dataset samples an unbalanced binary classification dataset in a 50/50 fashion"""

    def __init__(self,
                 meta_path,
                 img_dir,
                 transform: transforms.Compose,
                 target_transform: transforms.Compose,
                 attribution: str=None,
                 split: float=0.85,
                 train: bool=True):
        super().__init__()
        df = pd.read_csv(meta_path, low_memory=False)
        if attribution:
            df = df[df['attribution'] == attribution]
            
        self.df_positive = df[df["target"] == 1].reset_index()
        self.df_negative = df[df["target"] == 0].reset_index()
        self.file_names_positive = self.df_positive['isic_id'].values
        self.file_names_negative = self.df_negative['isic_id'].values
        self.targets_positive = self.df_positive['target'].values
        self.targets_negative = self.df_negative['target'].values
        
        self.file_names_positive = slice_by_percentage(self.file_names_positive, split, train)
        self.file_names_negative = slice_by_percentage(self.file_names_negative, split, train)
        self.targets_positive = slice_by_percentage(self.targets_positive, split, train)
        self.targets_negative = slice_by_percentage(self.targets_negative, split, train)
            
        self.transform = transform
        self.target_transform = target_transform
        self.img_dir = img_dir
        self.train = train
        self.n_samples = (len(self.targets_positive) + len(self.targets_negative))

    def __len__(self):
        return len(self.file_names_positive) * 2 if self.train else self.n_samples

    def __getitem__(self, idx):
        # Determine the probability with which the 2 classes are picked
        if self.train:
            class_balance = 0.5
        else:
            class_balance = len(self.targets_negative) / self.n_samples
        if random.random() >= class_balance:
            df = self.df_positive
            file_names = self.file_names_positive
            targets = self.targets_positive
        else:
            df = self.df_negative
            file_names = self.file_names_negative
            targets = self.targets_negative
        # Make sure the idx fits the data (as we handle pos and neg in differents dfs)
        idx = idx % len(df.shape)
        # Get Image and target
        img_id = file_names[idx]
        img = default_loader(os.path.join(self.img_dir, img_id + '.jpg'))
        target = targets[idx]
        # Transform if configured
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
            
        return img, target, img_id


class DataHandler:

    def __init__(self,
                 data_dir: str = None,
                 meta_path: str = None,
                 train: bool=True,
                 batch_size = 'auto',
                 shuffle: bool = True,
                 num_workers = 1,
                 height: int = 450,
                 width: int = 600,
                 device: str = 'auto',
                 dtype: torch.dtype = torch.float32,
                 model_name: str = 'dense201',
                 attribution: str = None):
        # Set Params
        self.data_dir = data_dir
        self.meta_path = meta_path
        self.height = height
        self.width = width
        
        # Define Device 
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else
                                       'mps' if torch.backends.mps.is_available() else
                                       'cpu')
        else:
            assert (device in ['cuda', 'mps', 'cpu']), "Device needs to be cuda, mps, or cpu"
            self.device = torch.device(device)

        # Set Augmentations
        self.transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomResizedCrop(size=(self.height, self.width),
                                         scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=64,
                                   contrast=0.75,
                                   saturation=0.25,
                                   hue=0.04
                                   ),
            transforms.ConvertImageDtype(dtype),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        self.transform_val = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Resize((self.height, self.width)),
            transforms.ConvertImageDtype(dtype),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        self.target_transform = transforms.Compose([
                                 lambda x:torch.tensor(x), # or just torch.LongTensor
                                 lambda x:torch.nn.functional.one_hot(x,2).float()])
        
        self.data_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")), 'datasets', data_dir)
        # Metadata Version
        if self.meta_path:
            self.meta_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")), 'datasets', self.meta_path)
            self.dataset = CustomMetaDatasetBalanced(img_dir=self.data_dir,
                                                     meta_path=self.meta_path,
                                                     train=train,
                                                     transform=self.transform if train else self.transform_val,
                                                     target_transform=self.target_transform,
                                                     attribution=attribution)
        # ImageFolders Version
        else:
            self.dataset = CustomImgFolderDataset(root=self.data_dir,
                                                  transform=self.transform  if train else self.transform_val,
                                                  target_transform=self.target_transform)
    
            # Balancing: Get class weight
            n_bening = len(os.listdir(os.path.join(self.data_dir, 'bening')))
            n_malignant = len(os.listdir(os.path.join(self.data_dir, 'malignant')))
            n_total = n_bening + n_malignant
            self.class_weights = torch.as_tensor([n_malignant / n_total, n_bening / n_total])

        
        # Recommend num workers
        self.num_workers = num_workers
        self.batch_size = batch_size
        # Calculate batch size based on num workers
        if batch_size == 'auto' and num_workers != 'auto':
            self.batch_size = recommend_max_batch_size(image_folder=self.data_dir,
                                                       num_workers=num_workers, 
                                                       model_name=model_name, 
                                                       data_transforms=self.transform,
                                                       target_transforms=self.target_transform)
            print(f"Recommended Batch Size: {self.batch_size}")
        # Calculate num workers based on batch size
        if num_workers == 'auto' and batch_size != 'auto':
            self.num_workers = recommend_num_workers(image_folder=self.data_dir,
                                                     batch_size=batch_size, 
                                                     model_name=model_name, 
                                                     data_transforms=self.transform,
                                                     target_transforms=self.target_transform)
            print(f"Number of Recommended Workers: {self.num_workers}")
        # Set standard value if both are on auto
        elif num_workers == 'auto' and batch_size == 'auto':
            self.num_workers = 1
            self.batch_size = 32
            
        # Define Dataloader
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True if train else False,
                                     num_workers=self.num_workers,
                                     pin_memory=True)



if __name__ == "__main__":
    dh = DataHandler(data_dir='isic1920_fil_split/train/', num_workers=1)
    s, t, p = next(iter(dh.dataloader))
    print(t)
    
