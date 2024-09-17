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
                      local_ckeckpoint_path=local_ckeckpoint_path)
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
                    'attribution': attribution
                    }
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


def ssft(out_path: str = 'fine-tuning-data',
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

    models = [
        'resnet18/ISIC2024/train-image/image/Department of Dermatology, Hospital Clínic de Barcelona/20240917-111109/epoch=279-val_loss=0.20-val_accuracy=0.9998.ckpt',
        #'resnet34/ISIC2024/train-image/image/Department of Dermatology, Hospital Clínic de Barcelona/20240917-113254/epoch=279-val_loss=0.32-val_accuracy=0.9996.ckpt',
        #'resnet50/ISIC2024/train-image/image/Department of Dermatology, Hospital Clínic de Barcelona/20240917-115900/epoch=319-val_loss=0.11-val_accuracy=0.9996.ckpt',
        #'resnet152/ISIC2024/train-image/image/Department of Dermatology, Hospital Clínic de Barcelona/20240917-122913/epoch=199-val_loss=0.14-val_accuracy=0.9997.ckpt',
        #'densenet121/ISIC2024/train-image/image/Department of Dermatology, Hospital Clínic de Barcelona/20240917-131550/epoch=19-val_loss=0.02-val_accuracy=0.9994.ckpt',
        #'densenet161/ISIC2024/train-image/image/Department of Dermatology, Hospital Clínic de Barcelona/20240917-133332/epoch=79-val_loss=0.21-val_accuracy=0.9996.ckpt',
        'densenet169/ISIC2024/train-image/image/Department of Dermatology, Hospital Clínic de Barcelona/20240917-140901/epoch=159-val_loss=0.04-val_accuracy=0.9993.ckpt',
        'densenet201/ISIC2024/train-image/image/Department of Dermatology, Hospital Clínic de Barcelona/20240917-092802/epoch=279-val_loss=0.01-val_accuracy=0.9996.ckpt',
        #'tf_efficientnet_b0/ISIC2024/train-image/image/Department of Dermatology, Hospital Clínic de Barcelona/20240917-144144/',
        'vgg16/ISIC2024/train-image/image/Department of Dermatology, Hospital Clínic de Barcelona/20240917-183914/epoch=379-val_loss=0.01-val_accuracy=0.9997.ckpt',
        'inception_v3/ISIC2024/train-image/image/Department of Dermatology, Hospital Clínic de Barcelona/20240917-193502/epoch=199-val_loss=0.00-val_accuracy=0.9996.ckpt',
        #'xception71/ISIC2024/train-image/image/Department of Dermatology, Hospital Clínic de Barcelona/20240917-195936/epoch=119-val_loss=0.35-val_accuracy=0.9992.ckpt',
        #'mobilenetv2_140/ISIC2024/train-image/image/Department of Dermatology, Hospital Clínic de Barcelona/20240917-204319/epoch=219-val_loss=0.00-val_accuracy=1.0000.ckpt',
    ]

    ensemble = Ensemble(models=models)

    new_clinic = 'Memorial Sloan Kettering Cancer Center'



    dh = data.DataHandler(data_dir=dataset_name,
                          meta_path=meta_path,
                          batch_size=batch_size,
                          train=False,
                          width=width,
                          height=height,
                          num_workers=8,
                          model_name=model_name,
                          attribution=new_clinic)

    save_path = os.path.join(os.path.abspath(out_path), model_name, dataset_name, attribution)

    models_start_acc = ensemble.test_models(ds)
    ensemble_start_acc = ensemble.test_ensemble(ds)
    print(models_start_acc)
    print(ensemble_start_acc)
    exit()

    ensemble.classify_and_collect(dh, save_path)




    ssft_datasets = [
        "ahja",
    ]


if __name__ == "__main__":

    ssft()
    exit()
    
    models = ['resnet18',
              'resnet34',
              'resnet50',
              'resnet152',
              'densenet121',
              'densenet161',
              'densenet169',
              'densenet201',
              'tf_efficientnet_b0',
              'vgg16',
              'inception_v3',
              'xception71',
              'mobilenetv2_140',
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
