import os
import glob
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

from encoder import Encoder
from cfg import shift

class Calculator:
    def __init__(self):

        self.model = Encoder()

    def __call__(self, path: str, dataset=None):

        dataset = path if dataset is None else dataset
        y_pred = []
        y_test = []
        file_list = glob.glob(f'{path}{os.sep}*.json')
        progress_bar = tqdm(desc='Processing files', total=len(file_list), unit='file')
        for i, fname in enumerate(file_list):

        
            with open(fname, 'r') as f:
                data = json.load(f)

            img = np.asarray(Image.open(dataset + os.sep + data['imagePath']))  
            _ = self.model.inference(img)        
            y_pred += [self.model.n]
 
            x1 = data['shapes'][0]['points'][0][0]
            n = int(x1/shift%(len(self.model.cards)-1)) + 1
            y_test += [n-1]

            score = self.calc_score(y_test, y_pred)
            progress_bar.set_postfix({'Score': score})
            progress_bar.update(1)
            
        return score

    def calc_score(self, y_test, y_pred):
        accuracy = []
        for y, p in zip(y_test, y_pred):
            accuracy += [p == y]
        if not len(accuracy): return 0
        score = sum(accuracy)/len(accuracy)
        return score
    