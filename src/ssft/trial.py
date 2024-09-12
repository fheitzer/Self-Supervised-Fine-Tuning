import torch
import timm
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import data
from torchvision import datasets, transforms

import os
import time
from timm.optim import AdamP
from timm.loss import BinaryCrossEntropy

torch.set_float32_matmul_precision('medium')

import torch
import timm
from pytorch_lightning import LightningModule, Trainer, seed_everything
import numpy as np
from torch import nn, optim


WIDTH = 650
HEIGHT = 450


def create_model(model="resnet18", pretrained=False, global_pool="catavgmax", num_classes=2):
    """helper function to overwrite the default values of the timm library"""
    model = timm.create_model(model, num_classes=num_classes, pretrained=pretrained, global_pool=global_pool)
    return model


class LitResnet(LightningModule):
    """"""
    
    def __init__(self, learning_rate: float=0.0001, model: str='resnet18', pretrained=False, global_pool="catavgmax", num_classes=2, features_only=False):
        super().__init__()
        self.save_hyperparameters()
        #self.device = torch.device('cuda' if torch.cuda.is_available() else
         #                          'mps' if torch.backends.mps.is_available() else
          #                         'cpu')
        self.model = create_model(model=model,
                                  pretrained=pretrained,
                                  global_pool=global_pool,
                                  num_classes=num_classes)
        # New Pytorch recommended way to send model to cpu
        if self.model.fc.out_features != num_classes:
            print("Fresh output layer added!")
            self.model.fc = nn.Linear(512, num_classes)
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.out_activation = torch.nn.functional.softmax
        # Where used?
        self.learning_rate = learning_rate

    def forward(self, x):
        out = self.model(x)
        return out

    def predict(self, x):
        out = self.model(x)
        out = self.out_activation(out, dim=1)
        return out

    def training_step(self, batch, batch_idx):
        # Define training step for Trainer Module
        # Example
        samples, targets = batch
        outputs = self.forward(samples)
        loss = self.criterion(outputs, targets)
        accuracy = self.binary_accuracy(outputs, targets)
        batch_size = targets.size(dim=0)
        self.log('train_accuracy', accuracy, prog_bar=True, batch_size=batch_size)
        self.log('train_loss', loss, prog_bar=True, batch_size=batch_size)
        #return {'loss':loss,"training_accuracy": accuracy}
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        accuracy = self.binary_accuracy(outputs, targets)
        loss = self.criterion(outputs, targets)
        batch_size = targets.size(dim=0)
        self.log('val_accuracy', accuracy, batch_size=batch_size)
        self.log('val_loss', loss, batch_size=batch_size)
        #return {"val_loss":loss, "test_accuracy":accuracy}
        #return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        accuracy = self.binary_accuracy(outputs, targets)
        loss = self.criterion(outputs, targets)
        batch_size = targets.size(dim=0)
        self.log('test_accuracy', accuracy, batch_size=batch_size)
        self.log('test_loss', loss, batch_size=batch_size)
        #return {"test_loss":loss, "test_accuracy":accuracy}
        #return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def binary_accuracy(self, outputs, targets):
        probabilities = self.out_activation(outputs, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
        target_classes = torch.argmax(targets, dim=1)
        correct_results_sum = (predicted_classes == target_classes).sum().float()
        acc = correct_results_sum/targets.shape[0]
        return acc

def cifar10_transformer():
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    target_transform = transforms.Compose([
                                 lambda x:torch.tensor(x), # or just torch.LongTensor
                                 lambda x:torch.nn.functional.one_hot(x,10).float()])
    return transform_train, transform_test, target_transform


def train_model_trial(out_path: str='models', 
                model_name: str='resnet18', 
                dataset_name: str='isic1920_fil_split', 
                device: str='cuda'
               ):
    # timestmap
    t = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(out_path, model_name, dataset_name)
    # Get Dataset Eval and Train
    batch_size = 256
    train_kwargs = {'batch_size': batch_size}
    cifar10_transformer_train, cifar10_transformer_test, target_transformer = cifar10_transformer()
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('data', 
                                                                     train=True,
                                                                     download=True,
                                                                     transform=cifar10_transformer_train,
                                                               target_transform=target_transformer), **train_kwargs)
    
    test_kwargs = {'batch_size': batch_size}
    #test_kwargs.update(cuda_kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('data',
                                                                    train=False,
                                                                    transform=cifar10_transformer_test,
                                                               target_transform=target_transformer), **test_kwargs)
    
    # Get the model
    model = LitResnet(model=model_name, pretrained=True, num_classes=10)
    # Define the callbacks
    callbacks = [ModelCheckpoint(save_dir + f"/{t}/",
                                 monitor='val_loss',
                                 mode='min',
                                 filename='{epoch}-{val_loss:.2f}',
                                 save_weights_only=True,
                                 every_n_epochs=1,
                                 save_top_k=5,
                                 ),
                 EarlyStopping(monitor='val_loss',
                               mode='min',
                               min_delta=0.0,
                               patience=100)]

    logger = TensorBoardLogger(save_dir, t + "_logs")
    # Configuring the trainer
    trainer = Trainer(devices=1,
                      accelerator=device,
                      callbacks=callbacks,
                      max_epochs=-1,
                      logger=logger,
                      log_every_n_steps=32)
    trainer.fit(model,
                train_loader,
                test_loader)


if __name__ == "__main__":
    models = ['resnet18',
              'resnet34',
              'resnet50',
              'resnet152',
              'densenet121',
              'densenet161',
              'densenet169',
              'densenet201',
              'densenet264d']
    models = ['resnet18']
    for model_name in models:
        train_model_trial(model_name=model_name)
