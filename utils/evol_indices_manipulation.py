import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from utils.various_tools import *


#Compute the Spearman rho of each column of a df containing evol_indices for different models
#with respect to a reference column. Return a dataframe of one line containing the Spearman's.
def compute_spearman_rho_from_evol_score_df(df, ref_column):
    ref = np.array(df[ref_column])
    dict_rhos = {}
    for column_name, values in df.items():
        if column_name!=ref_column and column_name!='mutations':
            dict_rhos[column_name] = spearmanr(ref,values)[0]
    return pd.DataFrame(dict_rhos, index=[0])


def df_evol_indices_Spearman(validation_df, list_EVE_models, list_Mi3_models, 
                             offset, list_mutations_location, out_file_location):
    '''
    Return and save to file a dataframe of one line containing the Spearman rho's of the evolutionary indices 
    for different models as compared to the validation scores.
    First, the previously computed evolutionary scores for the EVE models are loaded and added to a df containing the validation scores.
    Then, the two-site models are loaded and the evolutionary scores are computed and added to the df.
    Finally, the Spearman rho between each column and the validation column is computed, saved and returned.
    Parameters:
    - validation_df: (pd.DataFrame) contains the experimental score for each mutant.
    - list_EVE_models: (list) specifies which previously computed evolutionary indices from EVE models are loaded. 
        of the form [["model_name_1", "path_to_evol_indices_1" ], ["model_name_2", "path_to_evol_indices_2" ], ... ]
    - list_Mi3_models: (list) specifies which 2-site models (as Two_site_model objects) are loaded. 
        of the form [["model_name_1", "path_to_model_1" ], ["model_name_2", "path_to_model_2" ], ... ]
    - offset: (int) the position within the focus protein of the first site in the MSA.
    - list_mutations_location: (string) location of the file containing the list of mutations as written by EVE.
    - out_file_location: (string) path to the output file containing the df with the computed Spearman's.
    '''
    for i_model, [model_name, model_location] in enumerate(list_EVE_models):   
        EVE_evol_indices_df = pd.read_csv(model_location) 
        EVE_evol_indices_df.drop('protein_name', axis=1, inplace=True)
        EVE_evol_indices_df.drop(0, axis=0, inplace=True)
        EVE_evol_indices_df['evol_indices'] = EVE_evol_indices_df['evol_indices'].apply(lambda x: -x)
        if i_model == 0:
            EVE_models_df = EVE_evol_indices_df.rename(columns={'evol_indices': model_name}).copy(deep=True)
        else:
            mask = EVE_evol_indices_df['mutations'].isin(EVE_models_df['mutations'])
            EVE_models_df.insert(1 + i_model, model_name,  EVE_evol_indices_df[mask]['evol_indices'].tolist(), True)
            
    mask_EVE = EVE_models_df['mutations'].isin(validation_df['mutations'])
    master_df = pd.merge(validation_df.copy(deep=True)[['mutations', 
                                               'independent',
                                               'EVMutation',
                                              'experiment_linear']],
                        EVE_models_df[mask_EVE],
                        on="mutations")
    
    list_mutations=pd.read_csv(list_mutations_location, header=0)
    Mi3_df = list_mutations.copy(deep=True)
    for i_model, [Mi3_name, Mi3_location] in enumerate(list_Mi3_models):
        Mi3_model = load_instance_from_file(Mi3_location)
        relative_energies = Mi3_model.compute_evol_indices_single_mutations(list_mutations, offset)
        Mi3_df.insert(1+i_model, Mi3_name, - relative_energies, True)    
    mask_Mi3 = Mi3_df['mutations'].isin(validation_df['mutations'])
    master_df = pd.merge(master_df, Mi3_df[mask_Mi3], on="mutations")
    
    final_df = compute_spearman_rho_from_evol_score_df(master_df, ref_column = 'experiment_linear')
    final_df.to_csv(out_file_location, index=False)
    return final_df