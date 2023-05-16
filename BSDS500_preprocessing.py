import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.io import loadmat

def get_paths():
    IMAGES_DIR = Path('BSDS500/images')
    train_image_folder = IMAGES_DIR / 'train'
    train_image_files = list(map(str, train_image_folder.glob('*.jpg')))

    val_image_folder = IMAGES_DIR / 'val'
    val_image_files = list(map(str, val_image_folder.glob('*.jpg')))

    test_image_folder = IMAGES_DIR / 'test'
    test_image_files = list(map(str, test_image_folder.glob('*.jpg')))

    ANNOTATION_DIR = Path('BSDS500/ground_truth')
    train_annotation_folder = ANNOTATION_DIR / 'train'
    train_annotation_files = list(map(str, train_annotation_folder.glob('*.mat')))

    val_annotation_folder = ANNOTATION_DIR / 'val'
    val_annotation_files = list(map(str, val_annotation_folder.glob('*.mat')))

    test_annotation_folder = ANNOTATION_DIR / 'test'
    test_annotation_files = list(map(str, test_annotation_folder.glob('*.mat')))
    return [sorted(train_image_files), sorted(val_image_files), sorted(test_image_files),
            sorted(train_annotation_files), sorted(val_annotation_files), sorted(test_annotation_files)]
            

# define the function to load the image and its corresponding annotation
def load_data(image_path, annotation_path):
    # load the image
    image = Image.open(image_path)
    # convert the image to a numpy array and normalize it
    image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    # regularize image
    if image.shape[0] != 321:
        image = tf.transpose(image, perm=[1, 0, 2])
    # trim the size a little bit for training
    image = image[1:, 1:, :]

    # load the annotation
    annotation = loadmat(annotation_path)['groundTruth'][0][0][0][0][1]
    # regularize annotation
    if annotation.shape[0] != 321:
        annotation = tf.transpose(annotation)
    # trim the size a little bit for training
    annotation = tf.convert_to_tensor([annotation[1:, 1:]])
    annotation = tf.transpose(annotation, perm = [1, 2, 0])
    # cast annotation data type to float32
    tf.cast(annotation, 'float32')

    return image, annotation
