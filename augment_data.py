import imgaug as ia
from imgaug import augmenters as iaa
import cv2
from pathlib import Path

seq = iaa.Sequential([
    iaa.Affine(scale={"x": (0.2, 3), "y": (0.2, 3)},rotate=(-5, 5)),
    iaa.Fliplr(0.5)],
    random_order=True)

def augment_data(path):
    basepath = path/"trainingData"
    AugtrainingPath = path/"Aug_trainingData"
    AugtrainingPath.mkdir(exist_ok=True)
    folder_in_basepath = basepath.iterdir()
    for item in folder_in_basepath:
        img_set = AugtrainingPath/item.name
        img_set.mkdir(exist_ok=True)
        i=0
        for file in item.iterdir():
            img = cv2.imread(str(file))
            image_aug = seq.augment_image(img)
            cv2.imwrite(f"{img_set}/image_{i}.jpg",image_aug)
            i+=1

path  = Path.cwd()
augment_data(path)