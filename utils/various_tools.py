import os
import subprocess
import pickle
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


alphabet = "ACDEFGHIKLMNPQRSTVWY"
alphabetg = "-ACDEFGHIKLMNPQRSTVWY"
q = len(alphabetg) #We consider the gap as an extra letter

aa_dict = {}
aag_dict = {}
for i,aa in enumerate(alphabet):
    aa_dict[aa] = i
for i,aa in enumerate(alphabetg):
    aag_dict[aa] = i


def shell_run(command):
    subprocess.run(command, shell=True)


def to_minutes(end_time, start_time):
    n_min = math.floor((end_time - start_time)//60)
    n_sec = math.floor((end_time - start_time)%60)
    if n_sec<10:
        additional = '0'
    else:
        additional = ''
    return str(n_min) + ":" + additional+ str(n_sec)
    

def timed(func):
    def wrapper():
        start_time = timer() 
        func()
        end_time = timer()
        print(f'Elapsed time: ' + to_minutes(end_time, start_time) + 'min')
    return wrapper


def np_load_dict(file_name):
    return np.load(file_name, allow_pickle=True).item()


def number_common_entries_dictionaries(dict_1, dict_2):
    common_keys = set(dict_1).intersection(dict_2) 
    return len(common_keys)


def save_instance_to_file(file_name,instance):
    with open(file_name, 'wb') as file:
        pickle.dump(instance, file)
    print(f'Object successfully saved to "{file_name}"')


def load_instance_from_file(file_name):
    with open(file_name, "rb") as file:
        loaded_instance = pickle.load(file)
    return loaded_instance


def plot_correlations(list_metrics):
    for metric in list_metrics:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(metric, interpolation='nearest')
        fig.colorbar(cax)
    plt.show()


def plot_histograms(list_metrics, bins = None):
    for metric in list_metrics:
        flat_metric = metric.reshape(-1)   
        plt.hist(flat_metric, bins=bins)
    plt.show() 


def mean_and_std(list_arrays):
    for array in list_arrays:
        print(f"[{np.mean(array):.4f}, {np.std(array):.4f}]")


#Given a matrix, set all components within 'neighbourhood' entries away from the diagonal to zero
def metric_tilde(metric,neighbourhood=5):
    return np.array([[0 if np.absolute(ii-jj)<neighbourhood else metric[ii,jj] 
                     for jj in range(metric.shape[0])] for ii in range(metric.shape[0])])


#Return the list of indices (i,j) corresponding to the components of a matrix 
#with magnitude in the 'top_percentage' top percentage.
def list_strong_couplings(metric, top_percentage=0.4):
    list_SC = []
    threshold = np.max(metric)*(1-top_percentage)
    for ii in range(metric.shape[0]):
        for jj in range(ii):
            if metric[ii,jj] > threshold:
                list_SC.append([ii,jj])
    return list_SC





