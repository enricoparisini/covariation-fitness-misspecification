import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

from utils.various_tools import *


def compare_tensors(t_1, t_2, plot = True, p_value = False, return_Q = False, linear_fit = False):   
    '''
    Function to quickly assess how similar two tensors are. Compute Pearson r, Spearman rho, average deviation, max deviation,
    and average relative deviation. It can also perform a linear fit between the entries of the tensors and make a plot. 
    Parameters:
    - t_1 and t_2: (np.array) the two tensors to be compared. They must have the same number of entries (not necessarily the same shape).
    - plot: (bool) plots entries of the tensors on a t_1 vs t_2 plane.
    - p_value: (bool) if True, displays the p_values of Pearson r and Spearman rho.
    - return_Q: (bool) if True, the list [Pearson r, Spearman rho, avg_deviation, max_deviation] is returned
    - linear_fit: (bool) if True, a linear fit is performed between the entries of the two tensors and the linear function is plotted 
        if the parameter plot is True.
    '''
    tensor_1, tensor_2 = np.array(t_1).flatten(), np.array(t_2).flatten()
    tensor_1_mean = np.mean(np.abs(tensor_1))
    pear_r, spear_r = pearsonr(tensor_1, tensor_2),spearmanr(tensor_1, tensor_2)
    avg_dev = np.sum(np.abs(tensor_1 - tensor_2))/tensor_1.shape[0]
    max_dev = np.max(np.abs(tensor_1 - tensor_2))
    avg_rel_dev = np.sum(np.abs((tensor_2 - tensor_1)/tensor_1_mean))/tensor_1.shape[0]
    if p_value:
        print(f"Pearson_r = {pear_r[0]:.4f}, p_value = {pear_r[1]:.2E}")
        print(f"Spearman_r = {spear_r[0]:.4f}, p_value = {spear_r[1]:.2E}")
    else:
        print(f"Pearson_r = {pear_r[0]:.4f}")
        print(f"Spearman_r = {spear_r[0]:.4f}")
    print(f"Average_deviation = {avg_dev:.4f}")
    print(f"Max_deviation = {max_dev:.4f}") 
    print(f"Average_relative_deviation = {avg_rel_dev:.4f}")
    if linear_fit:
        z = np.polyfit(tensor_1, tensor_2, 1)
        print(f"Linear fit:  tensor_2 = {z[0]:.4f} tensor_1 + {z[1]:.4f}")
    if plot:
        _ , ax = plt.subplots()
        ax.scatter(tensor_1, tensor_2, s=5)
        max_value = np.max([tensor_1, tensor_2])*1.03
        min_value = np.min([tensor_1, tensor_2])-0.03*max_value  
        ax.plot([min_value, max_value], [min_value, max_value], color='black')
        if linear_fit:
            p = np.poly1d(z)
            min_1 = np.min(tensor_1)
            max_1 = np.max(tensor_1)
            x = np.arange(min_1, max_1, (max_1-min_1)/5)
            y = p(x)
            ax.plot(x, y, color='orange')
        plt.show()
    if return_Q:
        return np.array([pear_r[0], spear_r[0], avg_dev, max_dev])


def rs_observables_to_df(list_models, observables = ["f1", "f2", "CM2", "MI"], ref_label = "training", 
                         output_filename = None, return_Q = True,
                         correlations_folder='./results/correlations'):
    '''
    Given a list of Correlations objects, compare their marginals with those of a reference Correlation object within that list
    (either training or test Correlations object). The resulting Pearson r's are cast into a df, which is returned or saved to file.
    Parameters:
    - list_models: (list of Correlations objects) models that are to be compared. Each entry of the list is a pair of the form
        ["label", "name_model"], where "name_model" is the name of the model to upload from the correlations folder.
    - observables: (list of strings) set of marginals or correlations within the Correlations objects to we compare the models.
    - ref_label: (string) label that identifies the reference model.
    - output_filename: (string) path where the df is written to.
    - return_Q: (bool) if True, the df is returned.
    - correlations_folder: (string) folder location of Correlations files.
    '''     
    list_models_compare = [model  for model in list_models  if model[0]!=ref_label]
    list_models_compare_names = [model[0]  for model in list_models_compare]
    ref = [model  for model in list_models  if model[0]==ref_label][0]
    ref_model = load_instance_from_file(correlations_folder + os.sep + ref[1] + 
                                    os.sep + ref[1] + ".Correlations")
    rs_df = pd.DataFrame(np.nan, index=range(len(observables)), columns = ["observable"] + list_models_compare_names)
    rs_df["observable"] = observables
    
    for model_name, model_id in list_models_compare:
        model = load_instance_from_file(correlations_folder + os.sep + model_id + 
                                        os.sep + model_id + ".Correlations")
        rs_per_model = []
        for observable in observables:
            ref_tensor = eval(f"np.array(ref_model.{observable}).flatten()")
            tensor = eval(f"np.array(model.{observable}).flatten()")
            rs_per_model.append(pearsonr(ref_tensor, tensor)[0])
        rs_df[model_name] = rs_per_model

    if output_filename is not None:
        rs_df.to_csv(output_filename, index=False)
    if return_Q:
        return rs_df