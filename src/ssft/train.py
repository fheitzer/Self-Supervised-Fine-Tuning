import torch
import timm
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import data
from networks import LitResnet, Ensemble

import os
import time
from timm.optim import AdamP
from timm.loss import BinaryCrossEntropy

torch.set_float32_matmul_precision('medium')


def train_model(out_path: str='models', 
                model_name: str='resnet18', 
                dataset_name: str='isic1920_fil_split', 
                device: str='cuda', 
                batch_size: int=None,
                precision: int=32,
                num_workers: int=None
               ):
    # timestmap
    t = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(out_path, model_name, dataset_name)
    # Get Dataset Eval and Train
    dh_train = data.DataHandler(os.path.join(dataset_name, 'train/'),
                                batch_size=batch_size,
                                num_workers=num_workers,
                                model_name=model_name)
    dh_val = data.DataHandler(os.path.join(dataset_name, 'val/'),
                              batch_size=batch_size,
                              num_workers=num_workers,
                              model_name=model_name,
                              shuffle=False)
    # Get the model
    model = LitResnet(model=model_name)
    # Define the callbacks
    callbacks = [ModelCheckpoint(save_dir + f"/{t}/",
                                 monitor='val_loss',
                                 mode='min',
                                 filename='{epoch}-{val_loss:.2f}-{val_accuracy:.4f}',
                                 save_weights_only=True,
                                 every_n_epochs=1,
                                 save_top_k=3,
                                 ),
                 EarlyStopping(monitor='val_loss',
                               mode='min',
                               min_delta=0.0,
                               patience=50)]
    # Set TensorboardLogger
    logger = TensorBoardLogger(save_dir, t + "_logs")
    extra_params = {'batch_size': batch_size,
                    'num_workers': dh_train.num_workers,
                    'dataset_name': dataset_name,
                    'precision': precision}
    logger.log_hyperparams(extra_params)
    
    # Configuring the trainer
    trainer = Trainer(devices=1,
                      accelerator=device,
                      callbacks=callbacks,
                      max_epochs=-1,
                      logger=logger,
                      log_every_n_steps=32,
                      precision=precision)
    # Train
    trainer.fit(model,
                dh_train.dataloader,
                dh_val.dataloader)


if __name__ == "__main__":
    models = ['resnet18',
              'resnet34',
              'resnet50',
              'resnet152',
              'densenet121',
              'densenet161',
              'densenet169',
              'densenet201']
    #models = ['resnet18']
    for model_name in models:
        train_model(model_name=model_name,
                    batch_size=64, 
                    dataset_name='HAM10K_cli_split',
                    precision="16-mixed")
