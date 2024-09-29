import torch
import timm
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from timm.optim import AdamP
from timm.loss import BinaryCrossEntropy

import data
from networks import LitResnet, Ensemble
from utils import set_seed, save_dict
from plot import plot_cycle_accs, plot_cycle_data

import os
import gc
import time

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
    
    # Set TensorboardLogger
    #logger = TensorBoardLogger(save_dir, t + "_logs")
    #extra_params = {'batch_size': batch_size,
    #                'num_workers': dh_train.num_workers,
    #                'dataset_name': dataset_name,
    #                'meta_name': meta_name,
    #                'precision': precision,
    #                'attribution': attribution
    #                }
    #logger.log_hyperparams(extra_params)

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
    


def ssft(out_path: str = 'fine-tuning-data',
         device: str = 'cuda',
         batch_size: int = 256,
         precision = '16-mixed',
         num_workers: int = 0,
         height: int = 450,
         width: int = 600,
         local_ckeckpoint_path: str = None
         ):

    models = [
        #'densenet121/BCN/20240928-063408/epoch=61-val_loss=0.57-val_accuracy=0.7117-train_accuracy=0.5797.ckpt',
        #'densenet169/BCN/20240927-234735/epoch=77-val_loss=0.57-val_accuracy=0.7141-train_accuracy=0.5851.ckpt',
        #'densenet169/BCN/20240920-143313/epoch=14-val_loss=0.09-val_accuracy=1.0000.ckpt',
        #'densenet201/BCN/20240927-202356/epoch=91-val_loss=0.58-val_accuracy=0.7058-train_accuracy=0.5776.ckpt',
        #'inception_v3/BCN/20240921-142529/epoch=7-val_loss=0.02-val_accuracy=1.0000.ckpt',
        'resnet18/BCN/20240928-225505/epoch=231-val_loss=0.16-val_accuracy=0.9514-train_accuracy=0.9975.ckpt',
        'resnet34/BCN/20240929-011526/epoch=206-val_loss=0.14-val_accuracy=0.9715-train_accuracy=0.9981.ckpt',
        'resnet50/BCN/20240929-031832/epoch=215-val_loss=0.13-val_accuracy=0.9668-train_accuracy=0.9981.ckpt',
        #'resnet152/BCN/20240921-053140/epoch=72-val_loss=0.20-val_accuracy=1.0000.ckpt',
        'tf_efficientnet_b0/BCN/20240928-154653/epoch=225-val_loss=0.58-val_accuracy=0.9526-train_accuracy=0.9914.ckpt',
        #'tf_efficientnet_b1/BCN/20240928-181147/epoch=160-val_loss=0.74-val_accuracy=0.9312-train_accuracy=0.9891.ckpt',
        'tf_efficientnet_b2/BCN/20240928-201006/epoch=210-val_loss=0.62-val_accuracy=0.9478-train_accuracy=0.9958.ckpt',
        #'vgg16/BCN/20240921-112805/epoch=20-val_loss=0.32-val_accuracy=0.7642.ckpt',
    ]

    model_names = [x.split(r'/')[0] for x in models]

    new_clinics = [
                   #'VIENNA',
                   #'PH2',
                   #'UDA',
                   #'SIDNEY',
                   #'DERM7PT',
                   #'MSK',
                   #'HAM10000',
                   'BCN',
                  ]
    # Testing starting accuracy
    for clinic in new_clinics:
        t = time.strftime("%Y%m%d-%H%M%S")
        cycles = 10
        dh = data.DataHandler(data_dir=clinic,
                              meta_name=clinic,
                              batch_size=64,
                              train=False,
                              width=width,
                              height=height,
                              num_workers=0,
                              model_name=model_names,
                              attribution=clinic,
                              device=device,
                              split=None,
                              balanced=False
                             )
        print(f'Next clinic {clinic}')
        ensemble = Ensemble(models=models)
        ensemble.set_class_weights(dh.dataset.class_weights)

        collection_name = clinic + '_' + t
        # Take this to the loop
        for i in range(cycles):
            print(f'Next cycle {i}')
            ensemble.send_models_to_device()
            ensemble.convert_to_fp16()
            
            ensemble_start_accs = list()
            models_start_accs = list()
            
            print(f'Testing Clinic: {clinic}')
            ensemble_start_acc = ensemble.test_ensemble(dh.dataloader)
            ensemble_start_accs.append(ensemble_start_acc)
            print(ensemble_start_acc)
            models_start_acc = ensemble.test_models(dh.dataloader)
            models_start_accs.append(models_start_acc)
            print(models_start_acc)

            # Test the ensemble and models again
            accs_dict = dict(zip(model_names, [acc.item() for acc in models_start_acc]))
            accs_dict.update({'ensemble': ensemble_start_acc.item()})
            save_path = os.path.join(os.path.abspath(out_path), clinic, t, 'cycle_'+str(i))
            save_dict(accs_dict, name=save_path)
            plot_cycle_accs(os.path.join(os.path.abspath(out_path), clinic, t))
            
            save_cycle = os.path.join(collection_name, 'cycle_' + str(i))
            # Get Self Supervised Labels
            ensemble.classify_and_collect(dh, save_cycle)
            plot_cycle_data(save_cycle)
            # Perform one round of fine tuning
            ensemble.fine_tune_models(save_cycle, data_dir=clinic, only_classifier=False)

        del dh
        del ensemble
        torch.cuda.empty_cache()
        gc.collect()

    


if __name__ == "__main__":

    ssft()
    exit()
    
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
