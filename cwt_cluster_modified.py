# for loading/processing the images
import shutil

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

# models
from keras.applications.vgg16 import VGG16
from keras.models import Model

# clustering and dimension reduction
from keras.src.applications import VGG19, ResNet152V2, ResNet50
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
import time

def scan_filenames(path):

    # change the working directory to the path where the images are located
    os.chdir(path)

    # this list holds all the image filename
    filenames = []

    # creates a ScandirIterator aliased as files
    with os.scandir(path) as files:
        # loops through each file in the directory
        for file in files:
            if file.name.endswith('.png'): #and file.name.startswith('cwt'):
                # adds only the image files to the filenames list
                filenames.append(file.name)
    return filenames

# load the model first and pass as an argument

def extract_features(model, file):
    # load the image as a 224x224 array, changed to 1550x1540
    img = load_img(file, target_size=(224, 224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1, 224, 224, 3)
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features

def main(mode):
    model = VGG19()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)  # strip output layer

    path = f'C:/Users/Anders/OneDrive - USN/IIA/Semester 4 - Master Thesis/ProjectFiles/{mode}'
    if not os.path.exists(path+'/sort'):
        os.mkdir(path+'/sort')

    filenames = scan_filenames(path)

    data = {}

    #Get Statistics
    pred_times = []
    errors = 0
    correct = 0
    count = 0


    # lop through each image in the dataset
    for flower in filenames:
        count += 1
        start = time.time()
        # try to extract the features and update the dictionary
        path_flower = f'{path}/{flower}'
        feat = extract_features(model, path_flower)
        data[flower] = feat
        print("Test")
        end = time.time()
        if count < 1000:
            pred_times.append(end-start)

    print(f'Average prediction time: {sum(pred_times)/len(pred_times)}')
    print(f'Highes prediction time: {max(pred_times)}')
    print(f'Lowest prediction time: {min(pred_times)}')

    # get a list of the filenames
    filenames = np.array(list(data.keys()))

    # get a list of just the features
    feat = np.array(list(data.values()))

    # reshape so that there are 210 samples of 4096 vectors
    feat = feat.reshape(-1, 4096)

    # get the unique labels (from the flower_labels.csv)
    #df = pd.read_csv('flower_labels.csv')
    #label = df['label'].tolist()
    #unique_labels = list(set(label))



    #dimentionality reduciton by PCA #original 200, probably way too high
    pca = PCA(n_components=10, random_state=22)
    pca.fit(feat)
    x = pca.transform(feat)

    # cluster
    n_clusters = 7
    kmeans = KMeans(n_clusters=n_clusters,  random_state=22, algorithm='elkan')
    kmeans.fit(x)

    #sort
    for file, cluster in zip(filenames, kmeans.labels_):
        path_target_folder = f'{path}/sort/{cluster}'
        if not os.path.exists(path_target_folder):
            os.mkdir(path_target_folder)

        path_cwt = f'{path}/{file}'
        shutil.copy(path_cwt, path_target_folder)

        acc_file = file.replace("cwt_", "acc_")
        path_acc = f'{path}/{acc_file}'
        shutil.copy(path_acc, path_target_folder)


    df = pd.DataFrame([(a, b) for a, b in zip(filenames, kmeans.labels_)])
    df.columns = ['File', 'Class']
    df.to_csv(f'{path}/Zsummary_7.csv')

    #main('Friction')

if __name__ == '__main__':
    main('CWTCluster')