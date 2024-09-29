import torch
import PIL.Image
import os
import numpy as np
import random
from torchvision import transforms, datasets
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import gc

#torch.set_default_dtype(torch.float16)

pd.set_option("future.no_silent_downcasting", True)

# Stratified Kfold to split the data balanced by class and patient for train and val

from utils import recommend_num_workers, recommend_max_batch_size

from typing import Optional

def slice_by_percentage(x, percentage, train=True):
    split = int(len(x) * percentage)
    if train:
        return x[:split]
    else:
        return x[split:]

def preprocess_meta(df, meta_name, img_dir):
    # prepare metdata
    df = df[df['dx'].isin(['melanoma', 'nevus'])]
    if 'DERM7PT' not in meta_name:
        if 'PH2' in meta_name:
            df = df[df['dx_type'] == 'expert dermatologist']
        else:
            df = df[df['dx_type'] == 'histopathology']
    df = df[['image_id', 'dx']]
    df = df.rename(columns={'image_id':'isic_id', 'dx':'target'})
    df = df.replace({'melanoma':1, 'nevus':0})
    df_filenames = df['isic_id'].tolist()
    directory_filenames = os.listdir(img_dir)
    directory_filenames = [x[:-4] for x in directory_filenames]
    directory_filenames_set = set(directory_filenames)
    missing_files = [filename for filename in df_filenames if filename not in directory_filenames_set]
    df = df[~df['isic_id'].isin(missing_files)]
    return df


def compute_class_weights(targets):
    # Count the number of samples for each class
    class_0_count = np.sum(targets == 0)
    class_1_count = np.sum(targets == 1)
    
    # Total number of samples
    total_samples = len(targets)
    
    # Compute class weights as inverse of class frequency
    weight_0 = total_samples / (2 * class_0_count)
    weight_1 = total_samples / (2 * class_1_count)
    
    # Store the weights in a numpy array
    class_weights = np.array([weight_0, weight_1])
    return torch.from_numpy(class_weights)
    

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
                 meta_name,
                 img_dir,
                 transform: transforms.Compose,
                 target_transform: transforms.Compose,
                 attribution: str=None, # deprecated
                 split: float=0.85,
                 train: bool=True,
                 model_id: int=None,
                 filetype: str='.png',
                 process_meta: bool=True
                ):
        super().__init__()
        df = pd.read_csv(meta_name + '.csv', low_memory=False)
        self.empty = False
        if model_id:
            df = df[df['model_id'] == model_id]
        if process_meta:
            df = preprocess_meta(df, meta_name, img_dir)
        print(f'Dataset length: {df.shape[0]}')

        if df.shape[0] < 1:
            self.empty = True
            
        df = df.reset_index()
        df = df.sample(frac=1).reset_index(drop=True)
        self.file_names = df['isic_id'].values
        self.targets = df['target'].values

        if split:
            self.file_names = slice_by_percentage(self.file_names, split, train)
            self.targets = slice_by_percentage(self.targets, split, train)

        self.class_weights = compute_class_weights(self.targets)

        self.filetype = filetype
        self.transform = transform
        self.target_transform = target_transform
        self.img_dir = img_dir
        self.train = train
        del df
        gc.collect()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # Get Image and target
        img_id = self.file_names[idx]
        img = default_loader(os.path.join(self.img_dir, img_id + self.filetype))
        target = self.targets[idx]
        # Transform if configured
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
            
        return img, target, img_id

class CustomMetaDatasetBalanced(Dataset):
    """This dataset samples an unbalanced binary classification dataset in a 50/50 fashion"""

    def __init__(self,
                 meta_name,
                 img_dir,
                 transform: transforms.Compose,
                 target_transform: transforms.Compose,
                 attribution: str=None, # deprecated
                 split: float=0.85,
                 train: bool=True,
                 model_id: int=None,
                 filetype: str='.png',
                 process_meta: bool=True
                ):
        super().__init__()
        df = pd.read_csv(meta_name + '.csv', low_memory=False)
        self.empty = False
        if model_id:
            df = df[df['model_id'] == model_id]
        if process_meta:
            df = preprocess_meta(df, meta_name, img_dir)
        print(f'Dataset length: {df.shape[0]}')

        if df.shape[0] < 1:
            self.empty = True
        df = df.sample(frac=1).reset_index(drop=True)

        self.df_positive = df[df["target"] == 1].reset_index()
        self.df_negative = df[df["target"] == 0].reset_index()
        self.file_names_positive = self.df_positive['isic_id'].values
        self.file_names_negative = self.df_negative['isic_id'].values
        self.targets_positive = self.df_positive['target'].values
        self.targets_negative = self.df_negative['target'].values

        if split:
            self.file_names_positive = slice_by_percentage(self.file_names_positive, split, train)
            self.file_names_negative = slice_by_percentage(self.file_names_negative, split, train)
            self.targets_positive = slice_by_percentage(self.targets_positive, split, train)
            self.targets_negative = slice_by_percentage(self.targets_negative, split, train)
        if not self.empty:
            self.n_samples = (len(self.targets_positive) + len(self.targets_negative))
            print(f'Positive Targets: {len(self.targets_positive)}')
            print(f'Negative Targets: {len(self.targets_negative)}')
            print(f'Positive Target Percentage: {len(self.targets_positive) / self.n_samples}')
        self.filetype = filetype
        self.transform = transform
        self.target_transform = target_transform
        self.img_dir = img_dir
        self.train = train
        
        del df
        gc.collect()

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
        img = default_loader(os.path.join(self.img_dir, img_id + self.filetype))
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
                 meta_name: str = None,
                 train: bool=True,
                 batch_size = 'auto',
                 shuffle: bool = True,
                 num_workers = 1,
                 height: int = 450,
                 width: int = 600,
                 device: str = None,
                 dtype: torch.dtype = torch.float32,
                 model_name: str = 'dense201',
                 attribution: str = None,
                 model_id: int = None,
                 process_meta: bool=True,
                 split: float=0.85,
                 balanced: bool=True):
        # Set Params
        self.data_dir = data_dir
        self.meta_name = meta_name
        self.height = height
        self.width = width
        
        # Define Device 
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else
                                       'mps' if torch.backends.mps.is_available() else
                                       'cpu')

        # Set Augmentations
        self.transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.RandomResizedCrop(size=(self.height, self.width),
                                         scale=(0.8, 1.2),
                                         ratio=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2,
                                   contrast=0.2,
                                   saturation=0.2,
                                   hue=0.2
                                   ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ConvertImageDtype(dtype),
            # Imagenet Mean and Variance
            transforms.Normalize(mean=[0.485,0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.transform_val = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Resize((self.height, self.width)),
            transforms.ConvertImageDtype(dtype),
            # Imagenet Mean and Variance
            transforms.Normalize(mean=[0.485,0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.target_transform = transforms.Compose([
                                 lambda x:torch.tensor(x), # or just torch.LongTensor
                                 lambda x:torch.nn.functional.one_hot(x,2).float()])
        
        self.data_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")), 'datasets', data_dir)
        # Metadata Version
        if self.meta_name:
            self.meta_name = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")), 'datasets', 'metadata', self.meta_name)
            if balanced:
                self.dataset = CustomMetaDatasetBalanced(img_dir=self.data_dir,
                                                         meta_name=self.meta_name,
                                                         train=train,
                                                         transform=self.transform if train else self.transform_val,
                                                         target_transform=self.target_transform,
                                                         attribution=attribution,
                                                         model_id=model_id,
                                                         process_meta=process_meta,
                                                         split=split)
            else:
                self.dataset = CustomMetaDataset(img_dir=self.data_dir,
                                                         meta_name=self.meta_name,
                                                         train=train,
                                                         transform=self.transform if train else self.transform_val,
                                                         target_transform=self.target_transform,
                                                         attribution=attribution,
                                                         model_id=model_id,
                                                         process_meta=process_meta,
                                                         split=split)
                
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
    
