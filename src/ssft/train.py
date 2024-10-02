import torch
import timm
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torchmetrics
from timm.optim import AdamP
from timm.loss import BinaryCrossEntropy
import optuna

import data
from networks import LitResnet, Ensemble
from utils import set_seed, save_dict, save_top_trials
from plot import plot_cycle_accs, plot_cycle_data

import os
import gc
import time
from functools import partial

torch.set_float32_matmul_precision('medium')


def train_baseline_model(
                out_path: str = 'models',
                model_name: str = 'resnet18', 
                dataset_name: str = None,
                meta_name: str = None,
                device: str = 'cuda',
                batch_size: int = None,
                precision: int = 32,
                num_workers: int = 1,
                attribution: str = '',
                height: int = 450,
                width: int = 600,
                local_ckeckpoint_path: str = None,
                balanced: bool = False,
                class_weights: bool = True,
                pretrained: bool = False
               ):

    ### Reproducability
    set_seed(603853768)
    print(f'Model: {model_name}')
    print(f'Dataset: {dataset_name}')
    # timestamp
    t = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(out_path, model_name, dataset_name, attribution)
    # Get Dataset Eval and Train
    if meta_name:
        dh_train = data.DataHandler(data_dir=dataset_name,
                                    meta_name=meta_name,
                                    width=width,
                                    height=height,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    model_name=model_name,
                                    attribution=attribution,
                                    balanced=balanced,
                                   )
        dh_val = data.DataHandler(data_dir=dataset_name,
                                  meta_name=meta_name,
                                  batch_size=batch_size,
                                  train=False,
                                  width=width,
                                  height=height,
                                  num_workers=num_workers,
                                  model_name=model_name,
                                  attribution=attribution,
                                  balanced=False)
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
                      class_weights=dh_train.dataset.class_weights if class_weights else None,
                      local_ckeckpoint_path=local_ckeckpoint_path,
                      pretrained=pretrained)
    # Define the callbacks
    callbacks = [ModelCheckpoint(save_dir + f"/{t}/",
                                 monitor='val_loss',
                                 mode='min',
                                 filename='{epoch}-{val_loss:.2f}-{val_accuracy:.4f}-{train_accuracy:.4f}',
                                 save_weights_only=True,
                                 every_n_epochs=1,
                                 save_top_k=3,
                                 ),
                 EarlyStopping(monitor='val_loss',
                               mode='min',
                               min_delta=0.0,
                               patience=30)]

    # set logging frequency to every epoch
    batches_per_epoch = int(len(dh_train.dataset) / dh_train.batch_size)
    
    # Configuring the trainer
    trainer = Trainer(devices=1,
                      accelerator=device,
                      callbacks=callbacks,
                      max_epochs=-1,
                      #logger=logger,
                      enable_checkpointing=True,
                      log_every_n_steps=batches_per_epoch,
                      check_val_every_n_epoch=1,
                      precision=precision, # 16 or mixed precision doesnt work. loss gets nan
                      min_epochs=100,
                      default_root_dir=os.path.join(save_dir, t)
                      )
    extra_params = {'batch_size': batch_size,
                    'num_workers': dh_train.num_workers,
                    'dataset_name': dataset_name,
                    'meta_name': meta_name,
                    'precision': precision,
                    'attribution': attribution,
                    'balanced': balanced
                    }
    trainer.logger.log_hyperparams(extra_params)
    # Train
    trainer.fit(model,
                dh_train.dataloader,
                dh_val.dataloader)
    del dh_train
    del dh_val
    torch.cuda.empty_cache()
    gc.collect()
    

if __name__ == "__main__":
    models = [
              #'densenetblur121d',
              #'densenet201',
              #'densenet169',
              #'densenet161',
              #'densenet121',
              #'resnet152',
              'tf_efficientnet_b0',
              'tf_efficientnet_b1',
              'tf_efficientnet_b2',
              'resnet18',
              'resnet34',
              'resnet50',
              #'vgg16',
              #'inception_v3',
              #'xception71',
              #'mobilenetv2_140',
              ]
    
    for model_name in models:
        
        if 'eff' in model_name:
            if 'b0' in model_name:
                size = 224
            elif 'b1' in model_name:
                size = 240
            elif 'b2' in model_name:
                size = 260
        else:
            size = 224
            
        train_baseline_model(model_name=model_name,
                             local_ckeckpoint_path=None,
                             batch_size='auto',
                             num_workers=8,
                             dataset_name='BCN',
                             meta_name='BCN',
                             height=size,
                             width=size,
                             class_weights=True,
                             pretrained=True,
                             precision='16-mixed'
                             )
