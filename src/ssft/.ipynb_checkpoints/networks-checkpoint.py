import pandas as pd
import torch
import torchmetrics
import timm
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from torch import nn, optim
import os
import gc
from tqdm import tqdm

from torchmetrics.classification import BinaryConfusionMatrix

from PIL import Image
import data
from utils import save_dict


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
                 class_weights=None,
                 pretrained: bool = False):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model
        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.as_tensor(class_weights)
         # Set image input size
        if 'eff' in model:
            if 'b0' in model:
                self.size = 224
            elif 'b1' in model:
                self.size = 240
            elif 'b2' in model:
                self.size = 260
        else:
            self.size = 224
            
        if model == 'vit_base_patch16_224':
            global_pool = 'avg'
        
        if local_ckeckpoint_path and isinstance(local_ckeckpoint_path, str):
            local_ckeckpoint_path = {'file': local_ckeckpoint_path}
        self.model = timm.create_model(model,
                                       pretrained=True if local_ckeckpoint_path else pretrained,
                                       pretrained_cfg_overlay=local_ckeckpoint_path,
                                       global_pool=global_pool,
                                       num_classes=num_classes,
                                      )
        
        # self.model = torch.compile(self.model)
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.num_classes = num_classes
        self.out_activation = torch.nn.functional.softmax
        # Where used?
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.confusion_matrix = BinaryConfusionMatrix()

    def forward(self, x):
        logits = self.model(x)
        return logits

    def predict(self, x):
        x = self.check_size(x)
        logits = self.model(x)
        pred = self.out_activation(logits, dim=1)
        return pred

    def check_size(self, x):
        # Current size of the image tensor (assumed to be [batch, channels, height, width])
        _, _, current_height, current_width = x.size()

        target_size = (self.size, self.size)

        # Only downsize if the current dimensions are different from the target size
        if (current_height, current_width) != target_size:
            x = torch.nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

    def training_step(self, batch, batch_idx):
        # Define training step for Trainer Module
        samples, targets, _ = batch
        outputs = self.forward(samples)
        loss = self.criterion(outputs, targets)
        batch_size = targets.size(dim=0)
        accuracy, balanced_accuracy, sensitivity, specificity = self.compute_metrics(targets, outputs)
        self.log('train_accuracy', accuracy, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_balanced_accuracy', balanced_accuracy, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_sensitivity', sensitivity, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_specificity', specificity, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets, _ = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, targets)
        batch_size = targets.size(dim=0)
        accuracy, balanced_accuracy, sensitivity, specificity = self.compute_metrics(targets, outputs)
        self.log('val_accuracy', accuracy, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=False)
        self.log('val_loss', loss, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=False)
        self.log('val_balanced_accuracy', balanced_accuracy, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=False)
        self.log('val_sensitivity', sensitivity, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=False)
        self.log('val_specificity', specificity, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=False)

    def test_step(self, batch, batch_idx):
        inputs, targets, _ = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, targets)
        batch_size = targets.size(dim=0)
        accuracy, balanced_accuracy, sensitivity, specificity = self.compute_metrics(targets, outputs)
        self.log('test_accuracy', accuracy, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_loss', loss, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_balanced_accuracy', balanced_accuracy, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=False)
        self.log('test_sensitivity', sensitivity, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=False)
        self.log('test_specificity', specificity, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=False)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def binary_accuracy(self, outputs, targets):
        probabilities = self.out_activation(outputs, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
        target_classes = torch.argmax(targets, dim=1)
        correct_results_sum = (predicted_classes == target_classes).sum().float()
        acc = correct_results_sum/targets.shape[0]
        return acc

    def compute_metrics(self, targets, outputs):
        # Get class predictions and binary target
        probabilities = self.out_activation(outputs, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
        target_classes = torch.argmax(targets, dim=1)
        # Get confusion matrix
        conf_matrix = self.confusion_matrix(predicted_classes, target_classes)
        tn, fp, fn, tp = conf_matrix.flatten()
        # Convert these values to floats to prevent division issues
        tn, fp, fn, tp = tn.float(), fp.float(), fn.float(), tp.float()
        # Calculate specificity, sensitivity, balanced accuracy, and accuracy as tensors
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        balanced_accuracy = (sensitivity + specificity) / 2
        accuracy = (tp + tn) / (tp + fn + fp + tn)
        return accuracy, balanced_accuracy, sensitivity, specificity

    def freeze_feature_extractor(self):
        # Optionally freeze the feature extractor (backbone)
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze only the classifier layer (or specific layers)
        for param in self.model.get_classifier().parameters():
            param.requires_grad = True
        


class Ensemble(LightningModule):
    """
    One network to rule them all
    """

    def __init__(self,
                 models = None,
                 width: int=600,
                 height: int=450):
        super(Ensemble, self).__init__()
        self.models = list()
        self.model_names = list()
        if models and all(isinstance(model, LightningModule) for model in models):
            self.models = models
        elif models and all(isinstance(model, str) for model in models):
            self.model_paths = models
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
        self.width = width
        self.height = height
        self.low_precision = False

    def load_models(self, models):
        for model_path in models:
            model_name = model_path.split('/')[0]
            self.model_names.append(model_name)
            model_path = os.path.abspath(os.path.join('models', model_path))
            model = LitResnet.load_from_checkpoint(model_path)
            self.models.append(model)

    def unload_models(self):
        del self.models
        torch.cuda.empty_cache()
        gc.collect()
        self.models = list()
        
    def switch_models_to_cpu(self, keep_model_idx):
        for i, model in enumerate(self.models):
            if i != keep_model_idx:
                # Move model to CPU
                self.models[i] = model.to('cpu')
                self.models[i].confusion_matrix = model.confusion_matrix.to('cpu')
            else:
                self.models[i] = model.to('cuda')
                self.models[i].confusion_matrix = model.confusion_matrix.to('cuda')

    def send_models_to_device(self):
        for i, model in enumerate(self.models):
            self.models[i] = model.to(self.device_to_be)
            self.models[i].confusion_matrix = model.confusion_matrix.to('cuda')

    def convert_to_fp16(self):
        """Convert all models in the list to FP16 (half precision)."""
        for i, model in enumerate(self.models):
            # Convert model to FP16 precision
            self.models[i] = model.half()
        self.low_precision = True

    def convert_to_fp32(self):
        """Convert all models in the list to FP16 (half precision)."""
        for i, model in enumerate(self.models):
            # Convert model to FP16 precision
            self.models[i] = model.float()
        self.low_precision = False
    
    def set_learning_rate(self, lr):
        for model in self.models:
            model.learning_rate = lr

    def freeze_feature_extractors(self):
        for model in self.models:
            model.freeze_feature_extractor()

    def set_class_weights(self, class_weights):
        for model in self.models:
            model.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)


    def __call__(self,
             img,
             target=None,
             isic_id=None,
             collection_name: str = None,
             voting: str = 'hard',
             onehot: bool = True,
             return_model_preds=False,
             single_minority_vote_only=False,
             ):
        """
        Make prediction using a majority voting
        """
        assert voting in ['soft', 'hard'], "Voting has to be hard or soft!"
    
        img = img.to(self.device_to_be)
        if target is not None:
            target = target.to(self.device_to_be)
        if self.low_precision:
            img = img.half()
            if target is not None:
                target = target.half()
        
        data_ids = list()
        data_targets = list()
        data_preds = list()
        data_modelids = list()
        
        # Get model predictions (this should return a tensor of model outputs)
        model_predictions = self.get_model_predictions(img)  # Expecting this to return a single tensor
        ensemble_predictions = []

        # TODO: Implement soft voting
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

                    # Skip if more than one model is in the minority vote
                    if single_minority_vote_only and len(minority_ids) != 1:
                        continue 
    
                    # Collect data if needed
                    if collection_name:
                        for model_id in minority_ids:
                            data_ids.append(isic_id[i])
                            data_targets.append(torch.argmax(target[i]).item() if target is not None else None)
                            # TODO: Append not as binary, but the probability to 'scale' learning rate
                            data_preds.append(torch.argmax(ensemble_predictions[-1]).item() if onehot else ensemble_predictions[-1].item())
                            data_modelids.append(model_id)
    
            # Save the collected data to CSV if `collect` is enabled
            if collection_name:
                collection_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")), 'datasets', 'fine-tuning-data', collection_name + '.csv')
                
                if not os.path.exists(os.path.dirname(collection_path)):
                    os.mkdir(os.path.dirname(collection_path))
                
                data_collection = pd.DataFrame.from_dict({'isic_id': data_ids,
                                                          'target_true': data_targets, 
                                                          'target': data_preds,
                                                          'model_id': data_modelids})
                data_collection.to_csv(collection_path, index=False, mode='a', header=not os.path.exists(collection_path))

        if return_model_preds:
            return torch.stack(ensemble_predictions).to(self.device_to_be), model_predictions
        else:
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
        if self.low_precision:
            data = data.half()
        return torch.stack([model.predict(data) for model in self.models])


    def test_ensembleOLD(self, ds):
        """
        Test the performance of the ensemble on a test dataset.
        Returns accuracy, balanced accuracy, specificity, and sensitivity.
        """
        print('Testing Ensemble...')
        
        total_samples = 0.0
    
        # Initialize confusion matrix metric for binary classification
        confusion_matrix = torchmetrics.classification.BinaryConfusionMatrix().to(self.device_to_be)
    
        for i, (img, target, isic_id) in enumerate(tqdm(ds)):
            preds = self(img, onehot=False)  # Ensemble predictions (non-one-hot)
            
            # Ensure target is on the same device as the predictions
            target = target.to(self.device_to_be)
            target_classes = torch.argmax(target, dim=1)
            
            # Update confusion matrix with predictions and targets
            confusion_matrix.update(preds, target_classes)
            
            # Accumulate total samples
            total_samples += target.shape[0]
        
        # Compute the confusion matrix
        tn, fp, fn, tp = confusion_matrix.compute().flatten()
    
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn)  # Recall
        specificity = tn / (tn + fp)
        balanced_accuracy = (sensitivity + specificity) / 2
    
        # Return all metrics
        results = {
            "ensemble accuracy": accuracy.item(),
            "ensemble balanced_accuracy": balanced_accuracy.item(),
            "ensemble specificity": specificity.item(),
            "ensemble sensitivity": sensitivity.item()
        }
        
        return results

    def test_ensemble(self, ds, test_models=False, collection_name=None, single_minority_vote_only=False):
        """
        Test the performance of the ensemble on a test dataset.
        Returns accuracy, balanced accuracy, specificity, and sensitivity.
        """
        print('Testing Ensemble...')
        
        total_samples = 0.0
    
        # Initialize confusion matrix metric for binary classification
        confusion_matrix = torchmetrics.classification.BinaryConfusionMatrix().to(self.device_to_be)
        if test_models:
            metrics = [torchmetrics.classification.BinaryConfusionMatrix().to(self.device_to_be) for _ in self.models]
    
        for i, (img, target, isic_id) in enumerate(tqdm(ds)):
            if test_models:
                preds, preds_models = self(img,
                                           target,
                                           isic_id,
                                           onehot=False,
                                           return_model_preds=True,
                                           collection_name=collection_name,
                                           single_minority_vote_only=single_minority_vote_only
                                          )  # Ensemble predictions (non-one-hot)
            else:
                preds = self(img, onehot=False, collection_name=collection_name) 
            # Ensure target is on the same device as the predictions
            target = target.to(self.device_to_be)
            target_classes = torch.argmax(target, dim=1)
            
            # Update confusion matrix with predictions and targets
            confusion_matrix.update(preds, target_classes)
            
            # Accumulate total samples
            total_samples += target.shape[0]
        
            # Collect confusion matrix for each model
            for j, model_probs in enumerate(preds_models):
                predicted_classes = torch.argmax(model_probs, dim=1)
                metrics[j].update(predicted_classes, target_classes)
        
        # Compute the confusion matrix
        tn, fp, fn, tp = confusion_matrix.compute().flatten()
    
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn)  # Recall
        specificity = tn / (tn + fp)
        balanced_accuracy = (sensitivity + specificity) / 2
    
        # Return all metrics
        results = {
            "ensemble accuracy": accuracy.item(),
            "ensemble balanced_accuracy": balanced_accuracy.item(),
            "ensemble specificity": specificity.item(),
            "ensemble sensitivity": sensitivity.item()
        }
        if test_models:
            # Compute metrics for each model
            results_models = dict()
            for i, metric in enumerate(metrics):
                tn, fp, fn, tp = metric.compute().view(-1)
                
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                balanced_accuracy = (sensitivity + specificity) / 2
                
                results.update({
                    f"{self.model_names[i]} accuracy": accuracy.item(),
                    f"{self.model_names[i]} balanced_accuracy": balanced_accuracy.item(),
                    f"{self.model_names[i]} specificity": specificity.item(),
                    f"{self.model_names[i]} sensitivity": sensitivity.item()
                })
            
            return results, results_models
        else:
            return results

    def test_models(self, ds):
        # Initialize metrics for each model
        metrics = [torchmetrics.classification.BinaryConfusionMatrix().to(self.device_to_be) for _ in self.models]
        
        print('Testing Models...')
        for i, (img, target, isic_id) in enumerate(tqdm(ds)):
            probabilities = self.get_model_predictions(img)
            target = target.to(self.device_to_be)
            target_classes = torch.argmax(target, dim=1)  # Assuming target is one-hot encoded
            
            # Collect confusion matrix for each model
            for j, model_probs in enumerate(probabilities):
                predicted_classes = torch.argmax(model_probs, dim=1)
                metrics[j].update(predicted_classes, target_classes)
        
        # Compute metrics for each model
        results = dict()
        for i, metric in enumerate(metrics):
            tn, fp, fn, tp = metric.compute().view(-1)
            
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            balanced_accuracy = (sensitivity + specificity) / 2
            
            results.update({
                f"{self.model_names[i]} accuracy": accuracy.item(),
                f"{self.model_names[i]} balanced_accuracy": balanced_accuracy.item(),
                f"{self.model_names[i]} specificity": specificity.item(),
                f"{self.model_names[i]} sensitivity": sensitivity.item()
            })
        
        return results

    def classify_and_collect(self, dataset, collection_name, voting='hard'):
        """
        Make predictions on a dataset and collect continuous training data
        """
        print('Collect Self Supervised labels...')
        for img, target, isic_id in tqdm(dataset.dataloader):
            _ = self(img, target, isic_id, collection_name=collection_name, voting=voting)
        
    def fine_tune_models(self,
                         collection_name,
                         data_dir='ISIC2024/train-image/image/',
                         learning_rate=1e-5,
                         only_classifier=False,
                         batch_size=8,
                         epochs=4,
                         transforms=False):
        """
        Fine Tune that fuckrrrrr
        """
        self.set_learning_rate(lr=learning_rate)
        if self.low_precision is True:
            self.convert_to_fp32()
        if only_classifier:
            self.freeze_feature_extractors()
        collection_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")), 'datasets', 'fine-tuning-data')
        meta_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")), 'datasets', 'fine-tuning-data', collection_name)
        print('Fine tuning models...')
        for model_id, model in enumerate(tqdm(self.models)):
            self.switch_models_to_cpu(keep_model_idx=model_id)
            dh_train = data.DataHandler(data_dir=data_dir,
                                        meta_name=meta_path,
                                        width=model.size,
                                        height=model.size,
                                        batch_size=batch_size,
                                        num_workers=8,
                                        model_name=self.model_names[model_id],
                                        model_id=model_id,
                                        process_meta=False,
                                        train=transforms,
                                        split=None,
                                        balanced=False)
            if dh_train.dataset.empty:
                print(f'No fine-tuning data collected for {self.model_names[model_id]}...')
                continue
            #logger = TensorBoardLogger(os.path.join('models', 'fine-tuned', self.model_names[model_id]), collection_name + "_logs")
            
            batches_per_epoch = int(len(dh_train.dataset) / batch_size)
            
            trainer = Trainer(devices=1,
                      accelerator='cuda',
                      max_epochs=epochs,
                      #logger=logger,
                      #log_every_n_steps=batches_per_epoch,
                      enable_checkpointing=False,
                      logger=False,
                      precision='16-mixed'
                      )
            # Train
            trainer.fit(model,
                        dh_train.dataloader)
            
            # Uncomment to save model at every cycle
            #trainer.save_checkpoint(os.path.join('models', 'fine-tuned', self.model_names[model_id], collection_name + '.ckpt'), weights_only=True)
            del dh_train
            gc.collect()
        self.send_models_to_device()
        
