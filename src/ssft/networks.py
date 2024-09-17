import pandas as pd
import torch
import timm
from pytorch_lightning import LightningModule, Trainer, seed_everything
import numpy as np
from torch import nn, optim

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
                 pretrained=False,
                 global_pool="catavgmax",
                 num_classes=2,
                 features_only=False,
                 class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model
        
        if local_ckeckpoint_path:
            local_ckeckpoint_path = {'file': local_ckeckpoint_path}
        self.model = timm.create_model(model,
                                       pretrained=pretrained,
                                       pretrained_cfg=local_ckeckpoint_path,
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
        #scheduler = self.lr_schedulers()
        #scheduler.step()
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
                 models: list[LightningModule]):
        super(Ensemble, self).__init__()
        if models and all(isinstance(model, LightningModule) for model in models):
            self.models = models
        elif models and all(isinstance(model, str) for model in models):
            self.load_models(models)
        else:
            raise Exception("Models must be specified as lightning.LightningModule list")
        self.models = models
        self.fine_tuning_data = None
        self.missed_data = None
        self.num_classes = models[0].num_classes
        self.acc = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else
                                   'mps' if torch.backends.mps.is_available() else
                                   'cpu')

    def load_models(self, models):
        for model in models:
            self.models.append(LitResnet(model=model,
                                         #class_weights=None if meta_path else dh_train.class_weights,
                                         local_ckeckpoint_path=local_ckeckpoint_path,
                                         pretrained=True if local_ckeckpoint_path else None)
                               )

    def __call__(self,
                 img,
                 target=None,
                 isic_id=None,
                 collection_path: str = None,
                 voting: str = 'hard',
                 onehot: bool = True
                 ):
        """
        Make prediction using a majority voting
        """
        assert voting in ['soft', 'hard'], "Voting has to be hard or soft!"
        assert type(collect) == bool, "Collect has to be a boolean!"

        data_collection = pd.DataFrame(columns=['isic_id', 'target', 'prediction', 'model_idx'])
        
        model_predictions = self.get_model_predictions(img)
        ensemble_predictions = list()

        if voting == 'soft':
            # TODO: Implement soft vote
            # Get the average prediction for every datapoint and class. (Datapoint, Class)
            ensemble_predictions = np.mean(model_predictions, axis=0)

        if voting == 'hard':
            # Get the hard labels by each model. return shape = (Model, datapoint)
            hard_model_predictions = np.argmax(model_predictions, axis=-1)
            # Iterate over hard label list for each datapoint
            for i, datapoint_preds in enumerate(np.transpose(hard_model_predictions)):
                # Get the unique classes voted for and how often
                unique, counts = np.unique(datapoint_preds, return_counts=True, axis=-1)
                # Unanimous decision
                if len(unique) == 1:
                    # Make decision
                    if onehot:
                        ensemble_predictions.extend(np.eye(self.num_classes)[unique[0]])
                    else:
                        ensemble_predictions.extend(unique[0])
                    # Dont Collect data

                # Majority Vote
                if np.max(counts) >= (len(self.models) / 2):
                    # Make decision
                    majority_vote = unique[np.argmax(counts)]
                    if onehot:
                        ensemble_predictions.extend(np.eye(self.num_classes)[majority_vote])
                    else:
                        ensemble_predictions.extend(majority_vote)
                    minority_ids = [i for i, pred in enumerate(datapoint_preds) if pred != majority_vote]
                    # Collect data
                    if collect:
                        for model_id in minority_ids:
                            data_collection.append({'isic_id': isic_id,
                                                    'target': target,
                                                    'prediction': np.argmax(ensemble_predictions[-1]),
                                                    'model_idx': model_id,
                                                    },
                                                   ignore_index=True)

            data_collection.to_csv(collection_path, index=False, mode='a')

        return ensemble_predictions

    def fit(self, ds, epochs:int):
        """
        Fit the models of the ensemble to a dataset or to the continuous training data collected by the ensemble
        :param dataset:
        :return:
        """
        # For every model filter out relevant datapoints and fit it to them
        for i, model in enumerate(self.models):
            model.fit(self.fine_tuning_data.filter(lambda im, pred, model, lbl: model == i),
                      epochs=epochs)

    def get_model_predictions(self, data):
        """Return a list of the predictions of the individual models"""
        return np.asarray([model.predict(data).numpy() for model in self.models])

    def test_ensemble(self, ds):
        """
        Test the performance of the ensemble on a test dataset
        """
        acc_total = 0
        for i, img, target, isic_id in enumerate(ds):
            preds = self(img, onehot=False)
            # Collect accuracy
            target_classes = np.argmax(target, dim=1)
            correct_results_sum = (preds == target_classes).sum().float()
            acc = correct_results_sum / target.shape[0]
            acc_total = (acc_total + acc) / (i + 1)

        return acc_total

    def test_models(self, ds):
        """
        Test the performance of each model on a test dataset
        """
        accs_total = [0.0] * len(self.models)
        for i, img, target, isic_id in enumerate(ds):
            probabilities = self.get_model_predictions(img)
            target_classes = np.argmax(target, dim=1)
            # Collect accuracy
            for j, model_probs in enumerate(probabilities):
                predicted_classes = np.argmax(model_probs, dim=1)
                correct_results_sum = (predicted_classes == target_classes).sum().float()
                acc = correct_results_sum / targets.shape[0]
                accs_total[j] = (accs_total[j] + acc) / (i + 1)

        return accs_total

    def classify_and_collect(self, dataset, collection_path, voting='hard'):
        """
        Make predictions on a dataset and collect continuous training data
        """
        for img, target, isic_id in dataset:
            self(img, target, isic_id, collection_path=collection_path, voting=voting)

