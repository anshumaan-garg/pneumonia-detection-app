"""
Image preprocessing for the pneumonia detection app.

These steps MUST match the training pipeline exactly. A mismatch here degrades
predictions badly while raising no error, which makes it very hard to diagnose.

Training pipeline, in order:
  1. resize to 224 x 224 using INTER_AREA
  2. replicate the single greyscale channel to three
  3. apply the backbone's preprocess_input, on values still in the 0-255 range
"""
import cv2
import numpy as np

IMG_SIZE = 320


def load_backbone_preprocess(backbone_name):
    """Return the preprocess_input function matching the trained backbone."""
    from tensorflow.keras.applications import vgg16, resnet50, densenet
    return {
        "VGG16": vgg16.preprocess_input,
        "ResNet50": resnet50.preprocess_input,
        "DenseNet121": densenet.preprocess_input,
    }[backbone_name]


def prepare_image(image_array, backbone_name):
    """
    Convert a raw uploaded image into a batch ready for the model.

    image_array: HxW greyscale or HxWx3 colour, values 0-255
    returns:     (1, 224, 224, 3) float32, preprocessed for the backbone
    """
    # Uploaded files may be colour even when the content is greyscale - flatten first
    if image_array.ndim == 3 and image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    if image_array.ndim == 3 and image_array.shape[2] == 4:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2GRAY)

    # Step 1 - resize, matching training
    resized = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # Step 2 - greyscale to three channels
    rgb = cv2.cvtColor(resized.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Step 3 - backbone preprocessing, applied to 0-255 values (NOT pre-divided by 255)
    batch = np.expand_dims(rgb.astype(np.float32), axis=0)
    return load_backbone_preprocess(backbone_name)(batch)
