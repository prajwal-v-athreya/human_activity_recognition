# Importing required functions
from absl import logging
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import pandas as pd
import time
logging.set_verbosity(logging.ERROR)
import random
import re
import os
import tempfile
import ssl
import cv2
import numpy as np
import imageio
from IPython import display

from urllib import request  

# Utilities to open video files using CV2
def crop_center_square(frame):
  y, x = frame.shape[0:2]
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

def load_video(path, max_frames=0, resize=(224, 224)):
  cap = cv2.VideoCapture(path)
  frames = []
  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      frame = crop_center_square(frame)
      frame = cv2.resize(frame, resize)
      frame = frame[:, :, [2, 1, 0]]
      frames.append(frame)

      if len(frames) == max_frames:
        break
  finally:
    cap.release()
  return np.array(frames) / 255.0

# Loading the model using Tensorflow hub
i3d = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']

number = 1

def predict(sample_video):
  # Add a batch axis to the to the sample video.
  model_input = tf.constant(sample_video, dtype=tf.float32)[tf.newaxis, ...]

  logits = i3d(model_input)['default'][0]
  probabilities = tf.nn.softmax(logits)
  labels_csv = pd.read_csv(r'D:\projects\Nymble\kinetics_400_labels.csv')
  labels = labels_csv['name']
  print("For video number %d the top action is : " %number)
  for i in np.argsort(probabilities)[::-1][:1]:   #To get the top prediction on the human action in the video 
    print(f"  {labels[i]}: {probabilities[i] * 100:5.2f}%")
  number +=1


directory = r'D:\projects\Nymble\videos'   # Change path accoringly after downloading the zip folder.

for filename in os.listdir(directory):
    if filename.endswith(".mp4"):
      video_path = os.path.join(directory, filename)
    sample_video = load_video(video_path)
    predict(sample_video)