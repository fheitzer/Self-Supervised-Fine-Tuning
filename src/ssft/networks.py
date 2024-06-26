import tensorflow as tf
from tf.keras import Model
import torch
# from tensorflow.keras.applications.resnet import ResNet101, ResNet152
# from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169
import timm
import torch
import lightning as l
from pytorch_lightning import LightningModule, Trainer, seed_everything


WIDTH = 650
HEIGHT = 450


class Ensemble(Model):
    """
    One network to rule them all
    """

    def __init__(self,
                 models: List[torch.nn.Module] = None):
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

        model_predictions = np.asarray([model.predict(data).numpy() for model in self.models])
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
        img = tf.data.Dataset.from_tensor_slices(x)
        pred = tf.data.Dataset.from_tensor_slices(pred)
        datapoint = tf.data.Dataset.zip((img, pred))
        if self.fine_tuning_data is None:
            self.fine_tuning_data = datapoint
        else:
            self.fine_tuning_data = self.fine_tuning_data.concatenate(datapoint)

    def collect_miss(self, x, y):
        """Collect a datapoint which could not be determined.
        Review by hand later.
        """
        img = tf.data.Dataset.from_tensor_slices([x])
        if self.missed_data is None:
            self.missed_data = datapoint
        else:
            self.missed_data = self.missed_data.concatenate(datapoint)

    def load_data(self, filepath):
        self.set_continuous_training_data(
            tf.data.experimental.load(filepath[0],
                                      compression='GZIP',
                                      element_spec=self.continuous_data_spec))
        self.set_missed_data(
            tf.data.experimental.load(filepath[1],
                                      compression='GZIP',
                                      element_spec=self.missed_data_spec))

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


def create_model(model="resnet18", pretrained=False, global_pool="catavgmax", num_classes=2):
    """helper function to overwrite the default values of the timm library"""
    model = timm.create_model(model, num_classes=num_classes, pretrained=pretrained, global_pool=global_pool)
    return model


class LitResnet(LightningModule):
    def __init__(self, lr: float=0.05, model: str="resnet18", pretrained=False, global_pool="catavgmax", num_classes=2):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(model=model,
                                  pretrained=pretrained,
                                  global_pool=global_pool,
                                  num_classes=num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.nc = nc
        self.out_activation = torch.nn.functional.softmax
        # Where used?
        self.lr = lr

    def forward(self, x):
        out = self.model(x)
        out = self.out_activation(out)
        return out

    def training_step(self, batch, batch_idx):
        # Define training step for Trainer Module
        # Example
        x, y, z = batch
        out = self.model(x)
        loss = self.criterion(out, x)
        return loss


class CustomResNet50_2(tf.keras.models.Sequential):

    def __init__(self, input_shape=(WIDTH, HEIGHT, 3), pooling='max', downsampling=512, num_classes=2):
        """
        Initialize and configure the ResNet50 super class and add classifier to it matching the problem at hand
        :param input_shape: Image input shape. TF API states input width and height should not be smaller than 32.
        :param pooling: Pooling method to be applied after the last convolutional block
        """
        super(ResNet50_cet2, self).__init__()
        self.add(ResNet50(include_top=False,
                          weights=None,
                          input_shape=input_shape,
                          pooling=pooling))
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(downsampling))
        self.add(tf.keras.layers.Dense(num_classes, activation='softmax'))


class CustomResNet50(ResNet50):
    def __init__(self, input_shape=(WIDTH, HEIGHT, 3), pooling='max'):
        """
        Initialize and configure the ResNet50 super class and add classifier to it matching the problem at hand
        :param input_shape: Image input shape. TF API states input width and height should not be smaller than 32.
        :param pooling: Pooling method to be applied after the last convolutional block
        """
        super(ResNet50_cet, self).__init__(include_top=False,
                                           weights=None,
                                           input_shape=input_shape,
                                           pooling=pooling)


class CustomResNet101(ResNet101):
    def __init__(self, input_shape=(WIDTH, HEIGHT, 3), pooling='max'):
        """
        Initialize and configure the ResNet101 super class and add classifier to it matching the problem at hand
        :param input_shape: Image input shape. TF API states input width and height should not be smaller than 32.
        :param pooling: Pooling method to be applied after the last convolutional block
        """
        super(ResNet101_cet, self).__init__(include_top=False,
                                            weights=None,
                                            input_shape=input_shape,
                                            pooling=pooling)


class CustomResNet152(ResNet152):
    def __init__(self, input_shape=(WIDTH, HEIGHT, 3), pooling='max'):
        """
        Initialize and configure the ResNet152 super class and add classifier to it matching the problem at hand
        :param input_shape: Image input shape. TF API states input width and height should not be smaller than 32.
        :param pooling: Pooling method to be applied after the last convolutional block
        """
        super(ResNet152_cet, self).__init__(include_top=False,
                                            weights=None,
                                            input_shape=input_shape,
                                            pooling=pooling)
