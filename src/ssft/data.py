import torch
import PIL.Image
import os
import random
from torchvision import transforms, datasets


class CustomDataSet(datasets.ImageFolder):

    def __init__(self, root: str, transform: transforms.Compose):
        super().__init__(root=root, transform=transform)

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
                 num_workers: int = 4,
                 height: int = 450,
                 width: int = 600):
        self.data_dir = data_dir
        self.height = height
        self.width = width
        self.device = torch.device('cuda' if torch.cuda.is_available() else
                                   'mps' if torch.backends.mps.is_available() else
                                   'cpu')

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
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        if "/" not in data_dir:
            self.data_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")), 'datasets', data_dir)
        else:
            assert not os.path.isdir(data_dir), "There is no such directory!"
            self.data_dir = data_dir

        self.dataset = CustomDataSet(root=self.data_dir, transform=self.transform)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=batch_size,
                                                      shuffle=shuffle,
                                                      num_workers=num_workers)

    def __iter__(self):
        assert self.data_dir, "Data Directory needs to be specified!"
        for i, data in enumerate(self.dataloader, 0):
            inputs, label, path = data
            inputs, label = inputs.to(self.device), label.to(self.device)
            yield inputs, label, path
        #####################################
        # if not self.paths:
        #     self.read_dir()
        # if self.shuffle:
        #     random.shuffle(self.paths)
        #
        # for batch in self.paths:
        #     images = list()
        #     for image_path in batch:
        #         # Load PIL Image
        #         image = PIL.Image.open(image_path)
        #         # Transform pil image to torch tensor and apply transformations
        #         image = self.transform(image)
        #         # Send tensor to device (GPU)
        #         image = image.to(self.device)
        #         images.append(image)
        #     yield images

    # def batch(self, n):
    #     assert self.batch, "Already batched!"
    #     # Flatten
    #     self.paths = [item for sublist in self.paths for item in sublist]
    #     # Batch
    #     self.paths = [self.paths[i: i+n] for i in range(0, len(self.paths), n)]
    #     # Set batched to true
    #     self.batched = True
    #
    # def prefetch(self, n):
    #     pass
    #
    # def cache(self, n):
    #     pass
    #
    # def shuffle(self):

    #     self.shuffle = True
    #
    # def set_data_paths(self, data_dir):
    #     self.data_dir = data_dir
    #
    # def read_dir(self):
    #     self.paths = [[os.path.join(self.data_dir, file)] for file in os.listdir(self.data_dir) if x.endswith('.png')]
    ####################################################


if __name__ == "__main__":
    dh = DataHandler(data_dir='PH2', num_workers=1)
    for input, label, path in dh:
        print(path)
        break
