import tensorflow as tf
from tf.keras import Model
from tensorflow.keras.applications.resnet import ResNet101, ResNet152
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169

WIDTH = 650
HEIGHT = 450


class Ensemble(Model):
    """

    """

    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models
        self.continuous_training_data = None
        self.missed_data = None
        self.num_classes = models[0].num_classes
        self.acc = None
        self.continuous_data_spec = (tf.TensorSpec(shape=(WIDTH, HEIGHT, 3), dtype=tf.float64),
                                     tf.TensorSpec(shape=(2,), dtype=tf.float32),
                                     tf.TensorSpec(shape=(), dtype=tf.int32),
                                     tf.TensorSpec(shape=(2,), dtype=tf.float32))
        self.missed_data_spec = (tf.TensorSpec(shape=(WIDTH, HEIGHT, 3), dtype=tf.float64),
                                 tf.TensorSpec(shape=(2,), dtype=tf.float32))

    def __call__(self, x):
        """
        Make prediction using a majority voting
        :param args:
        :param kwargs:
        :return:
        """
        predictions = list()
        for model in self.models:
            pred = np.argmax(model.predict(y))


        return prediction

    def fit(self, dataset=None):
        """
        Fit the models of the ensemble to a dataset or to the continuous training data collected by the ensemble
        :param dataset:
        :return:
        """
        if dataset is None:
            self.fit(self.continuous_training_data)
        pass

    def review_and_collect_data(self, dataset):
        """
        Make predictions on a dataset and collect continuous training data
        :param dataset:
        :return:
        """
        predictions, models = self(dataset, black_box=False)

        pass

    def cycle_offline(self, data, num_cycles):
        """
        Cycle over dataset. In each cycle review and collect data, and then fit the ensemble models on continuous
        training data. Repeat cycles and hope for performance increase on passed dataset
        :param data:
        :return:
        """
        for i in num_cycles:
            self.review_and_collect_data(data)
            self.fit()
            self.reset_data()

    def collect_continuous_training_data(self, x, pred, wrong_model, label):
        """Add the current datapoint to self.data
        with the index of the model that needs to be trained on that datapoint
        and the label predicted by the other networks.
        """
        img = tf.data.Dataset.from_tensor_slices(x)
        pred = tf.data.Dataset.from_tensor_slices([pred]).map(lambda x: tf.one_hot(x, self.num_classes))
        wrong_model = tf.data.Dataset.from_tensor_slices([wrong_model])
        label = tf.data.Dataset.from_tensor_slices(label)
        datapoint = tf.data.Dataset.zip((img, pred, wrong_model, label))
        if self.continuous_training_data is None:
            self.continuous_training_data = datapoint
        else:
            self.continuous_training_data = self.continuous_training_data.concatenate(datapoint)

    def collect_miss(self, x, y):
        """Collect a datapoint which could not be determined.
        Review by hand later.
        """
        img = tf.data.Dataset.from_tensor_slices([x])
        label = tf.data.Dataset.from_tensor_slices(y)
        datapoint = tf.data.Dataset.zip((img, label))
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
        self.continuous_training_data = None
        self.missed_data = None

    def get_continuous_training_data(self):
        return self.continuous_training_data

    def set_continuous_training_data(self, ds):
        self.continuous_training_data = ds

    def get_missed_data(self):
        return self.missed_data

    def set_missed_data(self, ds):
        self.missed_data = ds


class ResNet50_cet2(tf.keras.models.Sequential):

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


class ResNet50_cet(ResNet50):
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


class ResNet101_cet(ResNet101):
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


class ResNet152_cet(ResNet152):
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
