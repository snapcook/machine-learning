import numpy as np
import os
import shutil
import cv2
import io
import matplotlib.pyplot as plt

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import xml.etree.ElementTree as ET
import tensorflow as tf
import json
import pickle

assert tf.__version__.startswith('2')

classes = ['Tempe', 'Daging sapi', 'carrot', 'apple', 'banana', 'orange', 'egg', 'potato', 'chicken', 'cabbage']

######## Load train and validation dataset
train_data = object_detector.DataLoader.from_pascal_voc(
    "/content/all_dataset/train",
    "/content/all_dataset/train",
    classes
)

val_data = object_detector.DataLoader.from_pascal_voc(
    "/content/all_dataset/valid",
    "/content/all_dataset/valid",
    classes
)

print("There's {} images on train dataset".format(len(train_data)))
print("There's {} images on validation dataset".format(len(val_data)))
########

######## Get model spec
spec = model_spec.get('efficientdet_lite2')
spec.config.learning_rate = 0.1

print(spec.__dict__)
########

######## Train the model
model = object_detector.create(train_data, model_spec=spec, batch_size=20, train_whole_model=True, epochs=30, validation_data=val_data)

print(model.__dict__)
########

######## Making plot history of model
history = model.model.history

######## Plotting the loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

######## Save history file
filename = 'model_history.pkl'
with open(filename, 'wb') as file:
    pickle.dump(history.history, file)

######## Save loss graph
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('train_loss.png')

######## Export model
model.export(export_dir='.', export_format=[ExportFormat.TFLITE])

######## Evaluate model
print("\n======================================evaluation on val data======================================\n")
evaluation_valid_results = model.evaluate(val_data)
print(evaluation_valid_results)
print("\n======================================evaluation on val data======================================\n")