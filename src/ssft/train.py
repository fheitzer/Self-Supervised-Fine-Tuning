import torch
import timm
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import data
from networks import LitResnet, Ensemble
from utils import set_seed

import os
import time
from timm.optim import AdamP
from timm.loss import BinaryCrossEntropy

torch.set_float32_matmul_precision('medium')


def train_baseline_model(out_path: str = 'models',
                model_name: str = 'resnet18', 
                dataset_name: str = None,
                meta_path: str = None,
                device: str = 'cuda',
                batch_size: int = None,
                precision: int = 32,
                num_workers: int = 1,
                attribution: str = '',
                height: int = 384,
                width: int = 384,
                local_ckeckpoint_path: str = None
               ):

    ### Reproducability
    set_seed(603853768)
    # timestamp
    t = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(out_path, model_name, dataset_name, attribution)
    # Get Dataset Eval and Train
    if meta_path:
        dh_train = data.DataHandler(data_dir=dataset_name,
                                    meta_path=meta_path,
                                    width=width,
                                    height=height,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    model_name=model_name,
                                    attribution=attribution
                                   )
        dh_val = data.DataHandler(data_dir=dataset_name,
                                  meta_path=meta_path,
                                  batch_size=batch_size,
                                  train=False,
                                  width=width,
                                  height=height,
                                  num_workers=num_workers,
                                  model_name=model_name,
                                  shuffle=False,
                                  attribution=attribution)
    else:
        dh_train = data.DataHandler(data_dir=os.path.join(dataset_name, 'train/'),
                                    width=width,
                                    height=height,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    model_name=model_name,
                                    attribution=attribution)
        dh_val = data.DataHandler(data_dir=os.path.join(dataset_name, 'val/'),
                                  batch_size=batch_size,
                                  width=width,
                                  height=height,
                                  num_workers=num_workers,
                                  model_name=model_name,
                                  shuffle=False,
                                  attribution=attribution)
    
    # Get the model
    model = LitResnet(model=model_name,
                      class_weights=None if meta_path else dh_train.class_weights,
                      local_ckeckpoint_path=local_ckeckpoint_path,
                      pretrained=True if local_ckeckpoint_path else None)
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
                               patience=10)]
    
    # Set TensorboardLogger
    logger = TensorBoardLogger(save_dir, t + "_logs")
    extra_params = {'batch_size': batch_size,
                    'num_workers': dh_train.num_workers,
                    'dataset_name': dataset_name,
                    'precision': precision,
                    'attribution': attribution}
    logger.log_hyperparams(extra_params)

    # set logging frequency to every epoch
    batches_per_epoch = int(len(dh_train.dataset) / batch_size)
    
    # Configuring the trainer
    trainer = Trainer(devices=1,
                      accelerator=device,
                      callbacks=callbacks,
                      max_epochs=-1,
                      logger=logger,
                      log_every_n_steps=batches_per_epoch,
                      check_val_every_n_epoch=20,
                      #precision=precision, # 16 or mixed precision doesnt work. loss gets nan
                      min_epochs=100
                      )
    # Train
    trainer.fit(model,
                dh_train.dataloader,
                dh_val.dataloader)


def ssft():

    models = [
        'resnet18/ISIC2024/train-image/image/Department of Dermatology, Hospital Clínic de Barcelona/20240917-111109/',
        'resnet34/ISIC2024/train-image/image/Department of Dermatology, Hospital Clínic de Barcelona/20240917-113254/',
        'resnet50/ISIC2024/train-image/image/Department of Dermatology, Hospital Clínic de Barcelona/20240917-115900/',
        'resnet152/ISIC2024/train-image/image/Department of Dermatology, Hospital Clínic de Barcelona/20240917-122913/',
        'densenet121/ISIC2024/train-image/image/Department of Dermatology, Hospital Clínic de Barcelona/20240917-131550/',
        'densenet161/ISIC2024/train-image/image/Department of Dermatology, Hospital Clínic de Barcelona/20240917-133332/',
        'densenet169/ISIC2024/train-image/image/Department of Dermatology, Hospital Clínic de Barcelona/20240917-140901/',
        'densenet121/ISIC2024/train-image/image/Department of Dermatology, Hospital Clínic de Barcelona/20240917-131550/',
        'tf_efficientnet_b0/ISIC2024/train-image/image/Department of Dermatology, Hospital Clínic de Barcelona/20240917-144144/',
    ]

    ensemble = Ensemble(models)


if __name__ == "__main__":
    
    models = ['resnet18',
              'resnet34',
              'resnet50',
              'resnet152',
              'densenet121',
              'densenet161',
              'densenet169',
              #'densenet201',
              'tf_efficientnet_b0',
              'vgg16',
              'inception_v3',
              'xception71',
              'mobilenetv2_140',
              'vit_base_patch16_224',
              ]
    local_checkpoint_paths = []
    
    for model_name in models:
        train_baseline_model(model_name=model_name,
                             local_ckeckpoint_path=None,
                             batch_size=32,
                             num_workers='auto',
                             dataset_name='ISIC2024/train-image/image/',
                             meta_path='ISIC2024/train-metadata.csv',
                             attribution="Department of Dermatology, Hospital Clínic de Barcelona"
                             )
