import pandas as pd
import torch
import timm
from pytorch_lightning import LightningModule, Trainer, seed_everything
import numpy as np
from torch import nn, optim
import os
from tqdm import tqdm

from PIL import Image



class GeMPooling(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeMPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'


class LitResnet(LightningModule):
    """"""
    
    def __init__(self,
                 learning_rate: float=1e-4,
                 weight_decay: float=1e-6,
                 model: str='tf_efficientnet_b0',
                 local_ckeckpoint_path: str = None,
                 global_pool="catavgmax",
                 num_classes=2,
                 class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model

        if model == 'vit_base_patch16_224':
            global_pool = 'avg'
        
        if local_ckeckpoint_path and isinstance(local_ckeckpoint_path, str):
            local_ckeckpoint_path = {'file': local_ckeckpoint_path}
        self.model = timm.create_model(model,
                                       pretrained=True if local_ckeckpoint_path else False,
                                       pretrained_cfg_overlay=local_ckeckpoint_path,
                                       global_pool=global_pool,
                                       num_classes=num_classes,
                                      )
        #in_features = self.model.classifier.in_features
        #self.model.classifier = nn.Identity()
        #self.model.global_pool = nn.Identity()
        #self.pooling = GeMPooling()
        #self.linear = nn.Linear(in_features, num_classes)
        #self.softmax = nn.Softmax()
        
        # self.model = torch.compile(self.model)
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.num_classes = num_classes
        self.out_activation = torch.nn.functional.softmax
        # Where used?
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x):
        logits = self.model(x)
        #pooled_features = self.pooling(features).flatten(1)
        #output = self.softmax(self.linear(pooled_features))
        return logits

    def predict(self, x):
        logits = self.model(x)
        pred = self.out_activation(logits, dim=1)
        return pred

    def training_step(self, batch, batch_idx):
        # Define training step for Trainer Module
        samples, targets, _ = batch
        outputs = self.forward(samples)
        loss = self.criterion(outputs, targets)
        accuracy = self.binary_accuracy(outputs, targets)
        batch_size = targets.size(dim=0)
        self.log('train_accuracy', accuracy, prog_bar=True, batch_size=batch_size)
        self.log('train_loss', loss, prog_bar=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets, _ = batch
        outputs = self.forward(inputs)
        accuracy = self.binary_accuracy(outputs, targets)
        loss = self.criterion(outputs, targets)
        batch_size = targets.size(dim=0)
        self.log('val_accuracy', accuracy, batch_size=batch_size)
        self.log('val_loss', loss, batch_size=batch_size)

    def test_step(self, batch, batch_idx):
        inputs, targets, _ = batch
        outputs = self.forward(inputs)
        accuracy = self.binary_accuracy(outputs, targets)
        loss = self.criterion(outputs, targets)
        batch_size = targets.size(dim=0)
        self.log('test_accuracy', accuracy, batch_size=batch_size)
        self.log('test_loss', loss, batch_size=batch_size)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=500, 
                                                         eta_min=1e-6)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler,
                                 'interval': 'step',
                                 'frequency': 1}
               }

    def binary_accuracy(self, outputs, targets):
        probabilities = self.out_activation(outputs, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
        target_classes = torch.argmax(targets, dim=1)
        correct_results_sum = (predicted_classes == target_classes).sum().float()
        acc = correct_results_sum/targets.shape[0]
        return acc


class Ensemble(LightningModule):
    """
    One network to rule them all
    """

    def __init__(self,
                 models = None):
        super(Ensemble, self).__init__()
        self.models = list()
        if models and all(isinstance(model, LightningModule) for model in models):
            self.models = models
        elif models and all(isinstance(model, str) for model in models):
            self.load_models(models)
        else:
            raise Exception("Models must be specified as lightning.LightningModule list or str list giving checkpoint paths")
        self.device_to_be = torch.device('cuda' if torch.cuda.is_available() else
                                         'mps' if torch.backends.mps.is_available() else
                                         'cpu')
        self.send_models_to_device()

        
        
        self.fine_tuning_data = None
        self.missed_data = None
        self.num_classes = self.models[0].num_classes
        self.acc = None

    def load_models(self, models):
        for model_path in models:
            model_name = model_path.split('/')[0]
            model_path = os.path.abspath(os.path.join('models', model_path))
            model = LitResnet.load_from_checkpoint(model_path)
            self.models.append(model)

    def send_models_to_device(self):
        for model in self.models:
            model.to(self.device_to_be)

    def __call__(self,
             img,
             target=None,
             isic_id=None,
             collection_path: str = None,
             voting: str = 'hard',
             onehot: bool = True,
             collect: bool = False  # Adding collect as an argument
             ):
        """
        Make prediction using a majority voting
        """
        assert voting in ['soft', 'hard'], "Voting has to be hard or soft!"
    
        img = img.to(self.device_to_be)
        if target is not None:
            target = target.to(self.device_to_be)
        
        data_collection = pd.DataFrame(columns=['isic_id', 'target', 'prediction', 'model_idx'])
        
        # Get model predictions (this should return a tensor of model outputs)
        model_predictions = self.get_model_predictions(img)  # Expecting this to return a single tensor
        ensemble_predictions = []
    
        if voting == 'soft':
            # Average the model predictions (soft voting)
            ensemble_predictions = torch.mean(model_predictions, dim=0)
    
        if voting == 'hard':
            # Get hard labels from each model (shape: (Model, Datapoint))
            hard_model_predictions = torch.argmax(model_predictions, dim=-1)
            
            # Iterate over hard label list for each datapoint
            for i, datapoint_preds in enumerate(hard_model_predictions.T):  # Transpose to (Datapoint, Model)
                # Get the unique classes voted for and their counts
                unique, counts = torch.unique(datapoint_preds, return_counts=True)
                
                # Move `unique` and `counts` to the same device
                unique = unique.to(self.device_to_be)
                counts = counts.to(self.device_to_be)
    
                # Unanimous decision
                if len(unique) == 1:
                    if onehot:
                        ensemble_predictions.append(torch.eye(self.num_classes, device=self.device_to_be)[unique[0]])
                    else:
                        ensemble_predictions.append(unique[0])
    
                # Majority vote decision
                elif torch.max(counts) >= (len(self.models) / 2):
                    majority_vote = unique[torch.argmax(counts)]
                    
                    if onehot:
                        ensemble_predictions.append(torch.eye(self.num_classes, device=self.device_to_be)[majority_vote])
                    else:
                        ensemble_predictions.append(majority_vote)
    
                    # Identify minority vote model indices
                    minority_ids = [j for j, pred in enumerate(datapoint_preds) if pred != majority_vote]
    
                    # Collect data if needed
                    if collect:
                        for model_id in minority_ids:
                            data_collection = data_collection.append({
                                'isic_id': isic_id,
                                'target': target.item() if target is not None else None,
                                'prediction': torch.argmax(ensemble_predictions[-1]).item() if onehot else ensemble_predictions[-1].item(),
                                'model_idx': model_id,
                            }, ignore_index=True)
    
            # Save the collected data to CSV if `collect` is enabled
            if collect:
                collection_path = os.path.join(os.path.abspath('fine-tune-data'), collection_path)
                data_collection.to_csv(collection_path, index=False, mode='a')
    
        return torch.stack(ensemble_predictions).to(self.device_to_be)  # Ensure the final tensor is on the right device

    def fit(self, ds, epochs:int):
        """
        UNFINISHED
        """
        # For every model filter out relevant datapoints and fit it to them
        for i, model in enumerate(self.models):
            model.fit(self.fine_tuning_data.filter(lambda im, pred, model, lbl: model == i),
                      epochs=epochs)

    def get_model_predictions(self, data):
        """Return a list of the predictions of the individual models"""
        data = data.to(self.device_to_be)
        return torch.stack([model.predict(data) for model in self.models])

    def test_ensemble(self, ds):
        """
        Test the performance of the ensemble on a test dataset
        """
        print('Testing Ensemble...')
        total_correct = 0.0
        total_samples = 0.0
        
        for i, (img, target, isic_id) in enumerate(tqdm(ds)):
            preds = self(img, onehot=False)  # Ensemble predictions (non-one-hot)
            
            # Ensure target is on the same device as the predictions
            target = target.to(self.device_to_be)
            target_classes = torch.argmax(target, dim=1)
            
            # Calculate the number of correct predictions
            correct_results_sum = (preds == target_classes).sum().float()
            
            # Accumulate correct predictions and total number of samples
            total_correct += correct_results_sum
            total_samples += target.shape[0]
        
        # Final accuracy calculation (total correct / total samples)
        accuracy = total_correct / total_samples
    
        return accuracy
    
    def test_ensemble_old(self, ds):
        """
        Test the performance of the ensemble on a test dataset
        """
        print('Testing Ensemble...')
        acc_total = 0
        for i, (img, target, isic_id) in enumerate(tqdm(ds)):
            #print(f'Batch {i}...')
            preds = self(img, onehot=False)
            # Collect accuracy
            target = target.to(self.device_to_be)
            target_classes = torch.argmax(target, dim=1)
            correct_results_sum = (preds == target_classes).sum().float()
            acc = correct_results_sum / target.shape[0]
            acc_total = (acc_total + acc) / (i + 1)

        return acc_total

    def test_models(self, ds):
        accs_total = [0.0] * len(self.models)
        total_samples = 0
        
        print('Testing Models...')
        for i, (img, target, isic_id) in enumerate(tqdm(ds)):
            probabilities = self.get_model_predictions(img)
            target = target.to(self.device_to_be)
            target_classes = torch.argmax(target, dim=1)
            total_samples += target.shape[0]
            
            # Collect accuracy for each model
            for j, model_probs in enumerate(probabilities):
                predicted_classes = torch.argmax(model_probs, dim=1)
                correct_results_sum = (predicted_classes == target_classes).sum().float()
                accs_total[j] += correct_results_sum
        
        # Calculate average accuracy after looping over the dataset
        average_accs = [correct_count / total_samples for correct_count in accs_total]

        return average_accs

    def test_models_old(self, ds):
        """
        Test the performance of each model on a test dataset
        """
        print('Testing Models...')
        accs_total = [0.0] * len(self.models)
        for i, (img, target, isic_id) in enumerate(tqdm(ds)):
            #print(f'Batch {i}...')
            probabilities = self.get_model_predictions(img)
            target = target.to(self.device_to_be)
            target_classes = torch.argmax(target, dim=1)
            # Collect accuracy
            for j, model_probs in enumerate(probabilities):
                predicted_classes = torch.argmax(model_probs, dim=1)
                correct_results_sum = (predicted_classes == target_classes).sum().float()
                acc = correct_results_sum / target.shape[0]
                accs_total[j] = (accs_total[j] + acc) / (i + 1)

        return accs_total

    def classify_and_collect(self, dataset, collection_path, voting='hard'):
        """
        Make predictions on a dataset and collect continuous training data
        """
        for img, target, isic_id in tqdm(dataset):
            self(img, target, isic_id, collection_path=collection_path, voting=voting)
