import torch
import PIL.Image
import os
import random
from torchvision import transforms, datasets


class CustomDataSet(datasets.ImageFolder):

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


class DataHandler:

    def __init__(self,
                 data_dir: str,
                 batch_size: int = 128,
                 shuffle: bool = True,
                 num_workers: int = 16,
                 height: int = 450,
                 width: int = 600,
                 device: str = 'auto',
                 dtype=torch.float32):
        self.data_dir = data_dir
        self.height = height
        self.width = width
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else
                                       'mps' if torch.backends.mps.is_available() else
                                       'cpu')
        else:
            assert (device in ['cuda', 'mps', 'cpu']), "Device needs to be cuda, mps, or cpu"
            self.device = torch.device(device)

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
        self.target_transform = None
        self.target_transform = transforms.Compose([
                                 lambda x:torch.tensor(x), # or just torch.LongTensor
                                 lambda x:torch.nn.functional.one_hot(x,2).float()])
        self.data_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")), 'datasets', data_dir)

        self.dataset = CustomDataSet(root=self.data_dir,
                                     transform=self.transform,
                                     target_transform=self.target_transform)
        #self.dataset.data = self.dataset.data.to(self.device)
        #self.dataset.target = self.dataset.target.to(self.device)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=batch_size,
                                                      shuffle=shuffle,
                                                      num_workers=num_workers)



if __name__ == "__main__":
    dh = DataHandler(data_dir='isic1920_fil_split/train/', num_workers=1)
    s, t, p = next(iter(dh.dataloader))
    print(t)
    
