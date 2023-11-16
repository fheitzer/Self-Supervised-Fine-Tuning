import os
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd

WIDTH = 650
HEIGHT = 450


def filter_isic_train():
    feature_names = [x for x in os.listdir('data/Isic/Train') if not x.startswith('.')]
    idx = [4, 3, -1, 0, 2, 1, -2, -4, -3]
    feature_names = [feature_names[x] for x in idx]

    list_of_images = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk('data/Isic/Train')]
                      for val in sublist][1:]

    zuweisungen = [0, 1, 2, 3, 4, 6, 6, 505, 5]

    features = dict(zip(feature_names, zuweisungen))

    isic = np.zeros((2058, HEIGHT, WIDTH, 3))
    isic_y = list()

    i = 0
    for path in list_of_images:
        # Squamous is excluded as it is not part of the ham10k dataset
        if 'squamous' in path:
            continue
        for feature in features.keys():
            if feature in path:
                isic[i] = Image.open(path).resize((WIDTH, HEIGHT))
                isic_y.append(features[feature])
                i += 1
                continue

    isic_y = np.asarray(isic_y)

    print(np.unique(isic_y, return_counts=True))

    filter_nvml = (isic_y == 3) | (isic_y == 4)
    isic_y = isic_y[filter_nvml]
    isic = isic[filter_nvml]

    isic = isic / 255
    isic = tf.data.Dataset.from_tensor_slices(isic)
    isic = isic.map(lambda x: tf.cast(x, tf.float32))

    isic_y = tf.data.Dataset.from_tensor_slices(isic_y)
    isic_y = isic_y.map(lambda x: tf.one_hot(x, 2))

    ds = tf.data.Dataset.zip((isic, isic_y))
    ds.save('data/Isic/isic_tf', compression='GZIP')


def filter_isic_test():
    feature_names = [x for x in os.listdir('data/Isic/Train') if not x.startswith('.')]
    idx = [4, 3, -1, 0, 2, 1, -2, -4, -3]
    feature_names = [feature_names[x] for x in idx]

    zuweisungen = [0, 1, 2, 3, 4, 6, 6, 505, 5]
    features = dict(zip(feature_names, zuweisungen))

    list_of_images = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk('data/Isic/Test')] for
                      val in sublist][1:]

    isic = np.zeros((102, HEIGHT, WIDTH, 3))
    isic_y = list()

    i = 0
    for path in list_of_images:
        # Squamous is excluded as it is not part of the ham10k dataset
        if 'squamous' in path:
            continue
        for feature in features.keys():
            if feature in path:
                isic[i] = Image.open(path).resize((WIDTH, HEIGHT))
                isic_y.append(features[feature])
                i += 1
                continue

    isic_y = np.asarray(isic_y)

    filter_nvml = (isic_y == 3) | (isic_y == 4)
    isic_y = isic_y[filter_nvml]
    isic = isic[filter_nvml]

    isic = isic / 255
    isic = tf.data.Dataset.from_tensor_slices(isic)
    isic = isic.map(lambda x: tf.cast(x, tf.float32))

    isic_y = tf.data.Dataset.from_tensor_slices(isic_y)
    isic_y = isic_y.map(lambda x: tf.one_hot(x, 2))

    ds = tf.data.Dataset.zip((isic, isic_y))
    ds.save('data/Isic/isic_tf_test', compression='GZIP')


def filter_ham10k():
    ham = pd.read_csv('data/ham10k/hmnist_28_28_RGB.csv')
    ham_meta = pd.read_csv('data/ham10k/HAM10000_metadata.csv')
    # Filter only images labelled with histo quality
    ham = ham[ham_meta['dx_type'] == 'histo']
    # Filter for nevus and melanoma
    ham = ham[(ham['label'] == 3) | (ham['label'] == 4)]
    # Separate labels
    ham_y = ham.pop('label')
    ham_y = np.asarray(ham_y)

    # Reshape image data
    ham = ham.to_numpy()
    ham = np.reshape(ham, (len(ham), HEIGHT, WIDTH, 3))

    ham = ham / 255
    ham = tf.data.Dataset.from_tensor_slices(ham)
    ham = ham.map(lambda x: tf.cast(x, tf.float32))

    ham_y = tf.data.Dataset.from_tensor_slices(ham_y)
    ham_y = ham_y.map(lambda x: tf.one_hot(x, 2))

    ds = tf.data.Dataset.zip((ham, ham_y))
    ds.save('data/ham10k/ham10k_tf_nvmel', compression='GZIP')


def filter_isic2020_train(start=0, end=None, name="test"):
    # Set metadata directory
    meta_dir = "data/Isic_2020/Train/ISIC_2020_Training_GroundTruth.csv"
    # Load metadata, select labels
    isic_y = pd.read_csv(meta_dir)['diagnosis'].to_numpy()
    # Cut out selection
    isic_y = isic_y[start:end]

    # Create mask for the wanted classes
    class_mask = (isic_y == 'melanoma') | (isic_y == 'nevus')
    # Apply mask to labels
    isic_y = isic_y[class_mask]

    # Set image directory
    folder_dir = "data/Isic_2020/Train/input"
    # Load image dirs
    image_dirs = os.listdir(folder_dir)
    # Check if there are some unnecessary files inbetween
    image_dirs = [x for x in image_dirs if (x.endswith(".jpg"))]
    # Cut out selection
    image_dirs = image_dirs[start:end]
    # For every image directory check if it is wanted (In class mask).
    # If it is then add the folder dir in front and collect it.
    image_dirs = [folder_dir + '/' + image_dir for image_dir, wanted in zip(image_dirs, class_mask) if wanted]

    # Initialize image collector
    isic = np.zeros((sum(class_mask), HEIGHT, WIDTH, 3))
    # Collect images
    for i, image in enumerate(image_dirs):
        isic[i] = Image.open(image).resize((WIDTH, HEIGHT))
        os.system('clear')
        print(i)


    # Preprocess data and save it
    isic = isic / 255
    isic = tf.data.Dataset.from_tensor_slices(isic)
    isic = isic.map(lambda x: tf.cast(x, tf.float32))

    isic_y = [0 if x == 'melanoma' else 1 for x in isic_y]
    isic_y = tf.data.Dataset.from_tensor_slices(isic_y)
    isic_y = isic_y.map(lambda x: tf.one_hot(x, 2))

    ds = tf.data.Dataset.zip((isic, isic_y))
    ds.save('data/Isic_2020/Train/tf/' + name, compression='GZIP')


def filter_isic2020_test():
    folder_dir = "data/Isic_2020/Test/ISIC_2020_Test_Input"
    image_dirs = os.listdir(folder_dir)
    image_dirs = [x for x in image_dirs if (x.endswith(".jpg"))]

    isic = np.zeros((len(image_dirs), HEIGHT, WIDTH, 3))

    for i, image in enumerate(image_dirs):
        isic[i] = Image.open(image).resize((WIDTH, HEIGHT))

    meta_dir = "data/Isic_2020/Test/..." #TODO: Add Ground Truth
    isic_y = pd.read_csv(meta_dir)['diagnosis']

    class_mask = (isic_y == 3) | (isic_y == 4)

    isic = isic[class_mask]
    isic_y = isic_y[class_mask]

    isic = isic / 255
    isic = tf.data.Dataset.from_tensor_slices(isic)
    isic = isic.map(lambda x: tf.cast(x, tf.float32))

    isic_y = isic_y.astype(int)
    isic_y = tf.data.Dataset.from_tensor_slices(isic_y)
    isic_y = isic_y.map(lambda x: tf.one_hot(x, 2))

    ds = tf.data.Dataset.zip((isic, isic_y))
    ds.save('data/Isic_2020/Test/tf/' + name, compression='GZIP')
    del isic
    del isic_y
    del ds


def filter_isic2019_train(start=0, end=None, name="all"):
    # Set metadata directory
    meta_dir = "data/Isic_2019/Train/ISIC_2019_Training_GroundTruth.csv"
    # Load metadata
    isic_y = pd.read_csv(meta_dir)
    # Make every NV Marker a 2 instead of a 1
    isic_y['NV'] = [x + 1 if x == 1 else 0 for x in isic_y['NV']]
    # Add the columns together to have 0, 1(MEL), 2(NV) labels
    isic_y = isic_y['MEL'] + isic_y['NV']
    # Convert to numpy
    isic_y = isic_y.to_numpy()
    # Cut out selection
    isic_y = isic_y[start:end]
    # Get Boolean mask
    class_mask = [False if x == 0 else True for x in isic_y]

    # Apply mask to labels
    isic_y = isic_y[class_mask]

    # Set image directory
    folder_dir = "data/Isic_2019/Train/ISIC_2019_Training_Input"
    # Load image dirs
    image_dirs = os.listdir(folder_dir)
    # Check if there are some unnecessary files inbetween
    image_dirs = [x for x in image_dirs if (x.endswith(".jpg"))]
    # Cut out selection
    image_dirs = image_dirs[start:end]
    # For every image directory check if it is wanted (In class mask).
    # If it is then add the folder dir in front and collect it.
    image_dirs = [folder_dir + '/' + image_dir for image_dir, wanted in zip(image_dirs, class_mask) if wanted]

    # Initialize image collector
    isic = np.zeros((sum(class_mask), HEIGHT, WIDTH, 3), dtype=np.float32)
    # Collect images
    for i, image in enumerate(image_dirs):
        isic[i] = Image.open(image).resize((WIDTH, HEIGHT))
        os.system('clear')
        print(i)


    # Preprocess data and save it
    #isic = isic / 255
    isic = tf.data.Dataset.from_tensor_slices(isic)
    isic = isic.map(lambda x: tf.cast(x, tf.float32))

    isic_y -= 1
    #isic_y = [0 if x == 'melanoma' else 1 for x in isic_y]
    isic_y = tf.data.Dataset.from_tensor_slices(isic_y)
    isic_y = isic_y.map(lambda x: tf.one_hot(tf.cast(x, tf.int32), 2))

    ds = tf.data.Dataset.zip((isic, isic_y))
    ds.save('data/Isic_2019/Train/tf/' + name, compression='GZIP')


# Outdated
def filter_isic2020_iteratively(name='In_parts/'):
    isic_len = 33127

    for i, t in enumerate(range(0, 33000, 1000)):
        t2 = t + 1000
        if t == 33000:
            t2 = isic_len
        print(i)
        filter_isic2020_train(start=t, end=t2, name=name + str(i))


# Outdated
def concat_all_tfds():
    dir_path = 'data/Isic_2020/Train/tf/In_parts'
    dirs = [dir_path + '/' + x for x in os.listdir(dir_path) if not x.startswith('.')]

    ds = tf.data.Dataset.load(dirs[0], compression='GZIP')
    for ds_dir in dirs[1:]:
        ds_2 = tf.data.Dataset.load(ds_dir, compression='GZIP')
        ds = ds.concatenate(ds_2)
        del ds_2
    ds.save('data/Isic_2020/Train/tf/isic_all', compression='GZIP')


if __name__ == "__main__":
    filter_isic2019_train(start=15000, end=None, name='15k_to_end')
