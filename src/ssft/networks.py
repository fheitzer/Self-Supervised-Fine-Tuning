import torch
import timm
from pytorch_lightning import LightningModule, Trainer, seed_everything
import numpy as np
from torch import nn, optim

from PIL import Image


WIDTH = 650
HEIGHT = 450

# Stratified Kfold to split the data balanced by class and patient for train and val

def create_model(model="resnet18", pretrained=False, global_pool="catavgmax", num_classes=2):
    """helper function to overwrite the default values of the timm library"""
    model = timm.create_model(model, num_classes=num_classes, pretrained=pretrained, global_pool=global_pool)
    return model


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
                 #checkpoint_path: str='/kaggle/input/tf-efficientnet/pytorch/tf-efficientnet-b0/1/tf_efficientnet_b0_aa-827b6e33.pth',
                 pretrained=False,
                 global_pool="catavgmax",
                 num_classes=2,
                 features_only=False,
                 class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        #self.device = torch.device('cuda' if torch.cuda.is_available() else
         #                          'mps' if torch.backends.mps.is_available() else
          #                         'cpu')
        self.model = create_model(model=model,
                                  pretrained=pretrained,
                                  #checkpoint_path=checkpoint_path,
                                  global_pool=global_pool,
                                  num_classes=num_classes,
                                  #features_only=features_only
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


class Ensemble(torch.nn.Module):
    """
    One network to rule them all
    """

    def __init__(self,
                 models: list[torch.nn.Module] = None):
        assert models is not None, "Must be initialized with list of models"
        super(Ensemble, self).__init__()
        if models and all(isinstance(s, torch.nn.Module) for s in lis):
            self.models = models
        elif models and all(isinstance(s, str) for s in lis):
            self._build_models(models)
        else:
            raise Exception("Models must be specified as torch.nn.Module list or list of strings with model names")
        self.models = models
        self.fine_tuning_data = None
        self.missed_data = None
        self.num_classes = models[0].num_classes
        self.acc = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else
                                   'mps' if torch.backends.mps.is_available() else
                                   'cpu')

    def __call__(self,
                 data,
                 voting: str = 'hard',
                 collect: bool = False,
                 return_type: str = 'numpy'):
        """
        Make prediction using a majority voting
        :param args:
        :param kwargs:
        :return:
        """
        assert voting in ['soft', 'hard'], "Voting has to be hard or soft!"
        assert type(collect) == bool, "Collect has to be a boolean!"
        assert return_type in ['numpy', 'tensor', 'list'], "return_type must be numpy, tensor, or list!"

        model_predictions = np.asarray([model.forward(data).numpy() for model in self.models])
        ensemble_predictions = list()
        # TODO: Implement soft vote
        if voting == 'soft':
            # Get the average prediction for every datapoint and class. (Datapoint, Class)
            ensemble_predictions = np.mean(model_predictions, axis=0)
        if voting == 'hard':
            cllt_datapoints = list()
            cllt_labels = list()
            # Get the hard labels by each model. return shape = (Model, datapoint)
            hard_model_predictions = np.argmax(model_predictions, axis=-1)
            # Iterate over hard label list for each datapoint
            for i, datapoint_preds in enumerate(np.transpose(hard_model_predictions)):
                # Get the unique classes voted for and how often
                unique, counts = np.unique(datapoint_preds, return_counts=True, axis=-1)
                # Unanimous decision
                if len(unique) == 1:
                    # Make decision
                    ensemble_predictions.extend(np.eye(self.num_classes)[unique[0]])
                    # Dont Collect data

                # Majority Vote
                if np.max(counts) >= (len(self.models) / 2):
                    # Make decision
                    majority_vote = unique[np.argmax(counts)]
                    ensemble_predictions.extend(np.eye(self.num_classes)[majority_vote])
                    # Collect data
                    if collect:
                        cllt_datapoints.extend(x[i])
                        cllt_labels.extend(majority_vote)

                # No majority
                else:
                    # TODO: Make dummy decision

                    # Collect miss
                    self.collect_miss(data)

        if return_type == 'tensor':
            return torch.FloatTensor(ensemble_predictions).to(self.device)
        elif return_type == 'numpy':
            return np.asarray(ensemble_predictions)
        elif return_type == 'list':
            return ensemble_predictions

    def fit(self, epochs:int):
        """
        Fit the models of the ensemble to a dataset or to the continuous training data collected by the ensemble
        :param dataset:
        :return:
        """
        ###### DOESNT WORK AS WE OLY COLLECT image id and label #########
        assert dataset is not None, "self.fine_tuning_data is None." \
                                    " Review a dataset to collect datapoints!"
        # For every model filter out relevant datapoints and fit it to them
        for i, model in enumerate(self.models):
            model.fit(self.fine_tuning_data.filter(lambda im, pred, model, lbl: model == i),
                      epochs=epochs)

    def test_ensemble(self, test_data, test_labels):
        """
        Test the performance of the ensemble on a test dataset
        """
        pass

    def test_models(self, test_data, test_labels):
        """
        Test the performance of each model on a test dataset
        """
        pass

    def review_and_collect(self, dataset):
        """
        Make predictions on a dataset and collect continuous training data
        :param dataset:
        :return:
        """
        predictions, models = self(dataset, return_type='')
        pass

    def cycle_offline(self, data, num_cycles, epochs_per_cycle, test_data):
        """
        Cycle over dataset. In each cycle review and collect data, and then fit the ensemble models on continuous
        training data. Repeat cycles and hope for performance increase on passed dataset
        :param data:
        :return:
        """
        for i in num_cycles:
            if test:
                self.test(test_data)
            _ = self(data, collect=True)
            self.fit(epochs=epochs_per_cycle)
            self.reset_data()

    def collect_continuous_training_data(self, x, pred):
        """Add the current datapoint to self.data
        with the index of the model that needs to be trained on that datapoint
        and the label predicted by the other networks.
        """
        pass
        #img = tf.data.Dataset.from_tensor_slices(x)
        #pred = tf.data.Dataset.from_tensor_slices(pred)
        #datapoint = tf.data.Dataset.zip((img, pred))
        #if self.fine_tuning_data is None:
         #   self.fine_tuning_data = datapoint
        #else:
         #   self.fine_tuning_data = self.fine_tuning_data.concatenate(datapoint)

    def collect_miss(self, x, y):
        """Collect a datapoint which could not be determined.
        Review by hand later.
        """
        pass
        #img = tf.data.Dataset.from_tensor_slices([x])
        #if self.missed_data is None:
         #   self.missed_data = datapoint
        #else:
         #   self.missed_data = self.missed_data.concatenate(datapoint)

    def load_data(self, filepath):
        pass
        #self.set_continuous_training_data(
         #   tf.data.experimental.load(filepath[0],
          #                            compression='GZIP',
           #                           element_spec=self.continuous_data_spec))
        #self.set_missed_data(
         #   tf.data.experimental.load(filepath[1],
          #                            compression='GZIP',
           #                           element_spec=self.missed_data_spec))

    def reset_data(self):
        """Data should be reset after each posttraining."""
        self.fine_tuning_data = None
        self.missed_data = None

    def get_continuous_training_data(self):
        return self.fine_tuning_data

    def set_continuous_training_data(self, ds):
        self.fine_tuning_data = ds

    def get_missed_data(self):
        return self.missed_data

    def set_missed_data(self, ds):
        self.missed_data = ds

    def _build_models(self, num_classes):
        pass


