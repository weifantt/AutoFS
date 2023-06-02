import numpy as np
import pandas as pd


def ReadData(name):
    if name == 'wild':
        dataset = pd.read_csv('dataset/wild.csv') # FC
        rem = ['Id']
        dataset.drop(rem,axis=1,inplace=True)
        return dataset

    elif name == 'tic': # ICB
        dataset = pd.read_csv('dataset/tic.csv')
        return dataset

    elif name == 'spam': # Spambase
        dataset = pd.read_csv('dataset/spam.csv')
        return dataset
        
    elif name == 'clean': # Musk
        dataset = pd.read_csv('dataset/clean.csv')
        rem = ['0','1']
        dataset.drop(rem,axis=1,inplace=True)
        return dataset
    return None
