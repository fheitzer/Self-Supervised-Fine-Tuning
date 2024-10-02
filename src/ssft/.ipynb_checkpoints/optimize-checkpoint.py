import os
import time

import optuna
from functools import partial

from utils import save_top_trials

def optimize_hyperparams(new_clinic='VIENNA',
                         cycles=12,
                         n_trials=10,
                         n_top=5,
                         transforms=False):
    # Get timestamp
    t = time.strftime("%Y%m%d-%H%M%S")
    # Define Pruner
    pruner = optuna.pruners.MedianPruner()
    # Create optimizer
    study = optuna.create_study(direction='maximize',
                                pruner=pruner,
                                storage='sqlite:///./optuna_study.db',
                                study_name=f'{new_clinic}_c_{cycles}_n_{n_trials}_{t}',
                                load_if_exists=True)
    # Set default values
    objective_func = partial(ssft,
                             new_clinic = new_clinic,
                             cycles = cycles,
                             transforms=transforms
                            )
    # Find optimized hyperparams
    study.optimize(objective_func,
                   n_trials=n_trials,
                   timeout=None)
    # Print best params
    print("Best hyperparameters: ", study.best_params)
    print("Best accuracy: ", study.best_value)
    # Save top n best params
    save_path = os.path.join(os.path.abspath('fine-tuning-data'),
                             'optimization',
                             new_clinic,
                             t + f'_{cycles}cycles')
    os.makedirs(save_path, exist_ok=True)
    save_top_trials(study, n=n_top, filename=save_path)


def optimize_clinics():
    clinics =[
               #'VIENNA',
               #'PH2',
               #'UDA',
               #'SIDNEY',
               #'DERM7PT',
               #'MSK',
               #'HAM10000',
               'BCN',
              ]
    for clinic in clinics:
        optimize_hyperparams(new_clinic=clinic,
                             cycles=12,
                             n_trials=10,
                             n_top=10,
                             transforms=True
                            )

if __name__ == "__main__":
    optimize_clinics()
