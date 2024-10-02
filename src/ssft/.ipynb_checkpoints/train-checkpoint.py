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
    


def ssft(trial = None,
         out_path: str = 'fine-tuning-data',
         new_clinic = 'VIENNA',
         cycles: int = 1,
         height: int = 450,
         width: int = 600,
         device: str = 'cuda',
         transforms=False,                  
         ):

    if not isinstance(new_clinic, list):
        new_clinic = [new_clinic]

    models = [
        'resnet18/BCN/20240928-225505/epoch=231-val_loss=0.16-val_accuracy=0.9514-train_accuracy=0.9975.ckpt',
        'resnet34/BCN/20240929-011526/epoch=206-val_loss=0.14-val_accuracy=0.9715-train_accuracy=0.9981.ckpt',
        'resnet50/BCN/20240929-031832/epoch=215-val_loss=0.13-val_accuracy=0.9668-train_accuracy=0.9981.ckpt',
        #'resnet152/BCN/20240921-053140/epoch=72-val_loss=0.20-val_accuracy=1.0000.ckpt',
        'tf_efficientnet_b0/BCN/20240928-154653/epoch=225-val_loss=0.58-val_accuracy=0.9526-train_accuracy=0.9914.ckpt',
        #'tf_efficientnet_b1/BCN/20240928-181147/epoch=160-val_loss=0.74-val_accuracy=0.9312-train_accuracy=0.9891.ckpt',
        'tf_efficientnet_b2/BCN/20240928-201006/epoch=210-val_loss=0.62-val_accuracy=0.9478-train_accuracy=0.9958.ckpt',
    ]

    model_names = [x.split(r'/')[0] for x in models]
    
    if trial is not None:
        ft_batch_size = trial.suggest_int('batch_size', 1, 16, step=8)
        ft_learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
        ft_only_classifier = trial.suggest_categorical('freeze_feature_extractor', [True])#, False])
        ft_epochs = trial.suggest_int('epochs_per_cycle', 4, 24, step=4)
        
    else:
        ft_batch_size = 1
        ft_learning_rate = 1e-6
        ft_only_classifier = True
        ft_epochs = 1
    
    # Testing starting accuracy
    for clinic in new_clinic:
        t = time.strftime("%Y%m%d-%H%M%S")
        dh = data.DataHandler(data_dir=clinic,
                              meta_name=clinic,
                              batch_size=64,
                              train=False,
                              width=width,
                              height=height,
                              num_workers=0,
                              model_name=model_names,
                              device=device,
                              split=None,
                              balanced=False)
        print(f'Next clinic {clinic}')
        ensemble = Ensemble(models=models)
        ensemble.set_class_weights(dh.dataset.class_weights)

        collection_name = clinic + '_' + t
        # Take this to the loop
        for i in range(cycles):
            print(f'Next cycle {i}')
            # Prepare ensemble for efficient training
            ensemble.send_models_to_device()
            ensemble.convert_to_fp16()
            # Save config in name
            name = f'batchsize_{ft_batch_size}_lr_{ft_learning_rate}_epochs_{ft_epochs}_freeze_{ft_only_classifier}'
            
            # Testing models
            print(f'Testing Clinic: {clinic}')
            
            # Define save dir
            save_cycle = os.path.join(collection_name, 'cycle_' + str(i))
            # This tests ensemble, models, and collects fine tuning data
            ensemble_metrics, models_metrics = ensemble.test_ensemble(dh.dataloader,
                                                                      test_models=True,
                                                                      collection_name=save_cycle,
                                                                      single_minority_vote_only=True)
            # Save accuracies
            accs_dict = models_metrics | ensemble_metrics
            save_path = os.path.join(os.path.abspath(out_path), clinic, t, 'cycle_'+str(i))
            save_dict(accs_dict, name=save_path)
            # Plot accuracies
            plot_cycle_accs(os.path.join(os.path.abspath(out_path), clinic, t), name=name, data_name=clinic)
            
            # Get Self Supervised Labels
            # ensemble.classify_and_collect(dh, save_cycle)
            # Plot cycle data
            plot_cycle_data(save_cycle, name=name, data_name=clinic)
            # Perform one cycle of fine tuning
            ensemble.fine_tune_models(save_cycle,
                                      data_dir=clinic,
                                      only_classifier=ft_only_classifier,
                                      batch_size=ft_batch_size,
                                      learning_rate=ft_learning_rate,
                                      epochs=ft_epochs,
                                      transforms=transforms,
                                     )
            # Tell the study the accuracy 
            if trial is not None:
                trial.report(ensemble_metrics['ensemble balanced_accuracy'], i)
                # Prune this trial if accuracy is meh
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            # Test A Last Time
            # Define save dir
            save_cycle = os.path.join(collection_name, 'cycle_' + str(cycles))
            # This tests ensemble, models, and collects fine tuning data
            ensemble_metrics, models_metrics = ensemble.test_ensemble(dh.dataloader,
                                                                      test_models=True,
                                                                      collection_name=None,
                                                                      single_minority_vote_only=True)
            # Save accuracies
            accs_dict = models_metrics | ensemble_metrics
            save_path = os.path.join(os.path.abspath(out_path), clinic, t, 'cycle_'+str(i))
            save_dict(accs_dict, name=save_path)
            # Plot accuracies
            plot_cycle_accs(os.path.join(os.path.abspath(out_path), clinic, t), name=name, data_name=clinic)
            # Plot Data
            plot_cycle_data(save_cycle, name=name, data_name=clinic)

    del dh
    del ensemble
    torch.cuda.empty_cache()
    gc.collect()
    
    if trial is not None:
        return ensemble_metrics['ensemble balanced_accuracy']


if __name__ == "__main__":
    ssft(new_clinic=[
                   #'VIENNA',
                   #'PH2',
                   #'UDA',
                   #'SIDNEY',
                   'DERM7PT',
                   #'MSK',
                   #'HAM10000',
                   #'BCN',
                  ],
         cycles=1
        )
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
