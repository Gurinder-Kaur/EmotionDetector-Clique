import pandas as pd
import numpy as np
import cv2
from pathlib import Path

def get_data(path):
    dataset = pd.read_csv(str(path) + "/data/fer2013.csv")
    data = dataset[dataset["Usage"]=="Training"]
    trainingPath = path/"trainingData"
    trainingPath.mkdir(exist_ok=True)
    for i in data.index:
        img_set = trainingPath/str(dataset['emotion'][i])
        img_set.mkdir(exist_ok=True)
        img = np.array(dataset['pixels'][i].split(' '),'float32').reshape(48,48)
        cv2.imwrite(f"{img_set}/image_{i}.jpg",img)

path  = Path.cwd()
get_data(path)

