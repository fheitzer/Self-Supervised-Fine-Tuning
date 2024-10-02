import os
import time

import torch

import data
from networks import Ensemble
from utils import save_dict
from plot import plot_cycle_accs, plot_cycle_data, plot_data_barchart

def sseft(trial = None,
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
        ft_batch_size = 16
        ft_learning_rate = 5e-5
        ft_only_classifier = True
        ft_epochs = 12
    
    # Clinic after Clinic
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
        # Get fresh ensemble for new clinic
        ensemble = Ensemble(models=models)
        # Adjust loss with class weights
        ensemble.set_class_weights(dh.dataset.class_weights)

        # Set name for data collection
        collection_name = clinic + '_' + t
        # Take this to the loop
        for i in range(cycles):
            # Verbose
            print(f'Next cycle {i}')
            # Prepare ensemble for efficient training
            ensemble.send_models_to_device()
            ensemble.convert_to_fp16()
            # Save sseft config in name
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
            # Save ensemble and model metrics
            accs_dict = models_metrics | ensemble_metrics
            save_path = os.path.join(os.path.abspath(out_path), clinic, t, 'cycle_'+str(i))
            save_dict(accs_dict, name=save_path)
            # Plot ensemble and model metrics
            plot_cycle_accs(os.path.join(os.path.abspath(out_path), clinic, t), name=name, data_name=clinic)
            # Plot fine tuning data metrics
            plot_cycle_data(save_cycle, name=name, data_name=clinic)
            # Plot fine tuning data distribution
            plot_data_barchart(save_cycle)
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

        # Test final models and ensemble
        ## Define save dir
        save_cycle = os.path.join(collection_name, 'cycle_' + str(cycles))
        ## This tests ensemble, models, and collects fine tuning data
        ensemble_metrics, models_metrics = ensemble.test_ensemble(dh.dataloader,
                                                                  test_models=True,
                                                                  collection_name=None,
                                                                  single_minority_vote_only=True)
        ## Save model and ensemble metrics
        accs_dict = models_metrics | ensemble_metrics
        save_path = os.path.join(os.path.abspath(out_path), clinic, t, 'cycle_'+str(i))
        save_dict(accs_dict, name=save_path)
        ## Plot ensemble and model metrics
        plot_cycle_accs(os.path.join(os.path.abspath(out_path), clinic, t), name=name, data_name=clinic)
        ## Plot fine tuning data distribution
        plot_cycle_data(save_cycle, name=name, data_name=clinic)

    # Clean up cache
    del dh
    del ensemble
    torch.cuda.empty_cache()
    gc.collect()

    # Return accuracy for trial objective
    if trial is not None:
        return ensemble_metrics['ensemble balanced_accuracy']


if __name__ == "__main__":
    sseft(new_clinic=[
                   #'VIENNA',
                   #'PH2',
                   #'UDA',
                   #'SIDNEY',
                   #'DERM7PT',
                   'MSK',
                   #'HAM10000',
                   #'BCN',
                  ],
          cycles=31,
          transforms=True
        )
