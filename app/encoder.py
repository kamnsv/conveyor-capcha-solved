from tensorflow.keras.applications import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow as tf
from scipy.spatial import distance
import numpy as np

from cfg import path_model, shape, shift

class Encoder:

    def __init__(self):
        self.encoder = Sequential([
            tf.keras.layers.TFSMLayer(path_model, call_endpoint='serving_default')
        ])
    
    def preproc(self, img:np.array, shape=shape[:2]):
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, shape).numpy()
        img = xception.preprocess_input(img)
        return img

    def inference(self, img:np.array) -> np.array:
        self.cards = self.get_images(img)
        tensors = np.array([self.preproc(i) for i in self.cards])
        embeddings = self.encoder.predict(tensors, verbose=False)['embedding']
        vector = embeddings[0].flatten()
        distances = [distance.euclidean(vector, e.flatten()) for e in embeddings[1:]]
        self.n = np.argmin(distances)
        return (70 + self.n * shift, 60)
    
    def get_images(self, img: np.array) -> list:
        card = img[230:330,10:110,:]
        cards = [card]
        n = 0
        while img.shape[1] > n*shift:
            card = img[:shift,n*shift:n*shift + shift,:][20:120,20:120]
            cards += [card]
            n += 1
        return cards
    