import os
import tqdm
import numpy as np
from scipy.stats import pearsonr, spearmanr


'''
Functions that compute 3,4,5,6 and general n marginals for a list of Correlations objects,
at a set of num_set_indices random site positions, over all letters at those positions.
'''
def produce_f3s_1free(list_models, num_set_indices=1):
    n_models = len(list_models)
    L = list_models[0].L
    q = list_models[0].q
    set_indices = np.sort(np.random.randint(L, size=(num_set_indices, 2)))   
    list_f3s = np.zeros((n_models, num_set_indices, 1, q, 1, q, L, q)) 
    for i_model, model in enumerate(list_models):
        list_f3s[i_model] = [model.compute_f3_no_pseudocount_1free(indices) for indices in set_indices]        
    return list_f3s


def produce_f3s(list_models, num_set_indices=1):
    n_models = len(list_models)
    L = list_models[0].L
    q = list_models[0].q
    set_indices = np.sort(np.random.randint(L, size=(num_set_indices, 3)))   
    list_f3s = np.zeros((n_models, num_set_indices, 1, q, 1, q, 1, q)) 
    for i_model, model in enumerate(list_models):
        list_f3s[i_model] = [model.compute_f3_no_pseudocount(indices) for indices in set_indices]        
    return list_f3s


def produce_f4s(list_models, num_set_indices=1):
    n_models = len(list_models)
    L = list_models[0].L
    q = list_models[0].q
    set_indices = np.sort(np.random.randint(L, size=(num_set_indices, 4)))
    list_f4s = np.zeros((n_models, num_set_indices, 1, q, 1, q, 1, q, 1, q))
    for i_model, model in enumerate(list_models):
        list_f4s[i_model] = [model.compute_f4_no_pseudocount(indices) for indices in set_indices]   
    return list_f4s


def produce_f5s(list_models, num_set_indices=1):
    n_models = len(list_models)
    L = list_models[0].L
    q = list_models[0].q
    set_indices = np.sort(np.random.randint(L, size=(num_set_indices, 5)))
    list_f5s = np.zeros((n_models, num_set_indices, 1, q, 1, q, 1, q, 1, q, 1, q))   
    for i_model, model in enumerate(list_models):
        list_f5s[i_model] = [model.compute_f5_no_pseudocount(indices) for indices in set_indices]     
    return list_f5s


def produce_f6s(list_models, num_set_indices=1):
    n_models = len(list_models)
    L = list_models[0].L
    q = list_models[0].q
    set_indices = np.sort(np.random.randint(L, size=(num_set_indices, 6)))
    list_f6s = np.zeros((n_models, num_set_indices, 1, q, 1, q, 1, q, 1, q, 1, q, 1, q)) 
    for i_model, model in enumerate(list_models):
        list_f6s[i_model] = [model.compute_f6_no_pseudocount(indices) for indices in set_indices]     
    return list_f6s


def produce_fns(list_models, n, num_set_indices=1):
    n_models = len(list_models)
    L = list_models[0].L
    q = list_models[0].q
    set_indices = np.sort(np.random.randint(L, size=(num_set_indices, n)))
    
    shape_list = q * np.ones(2 + 2*n, dtype = int)
    shape_list[::2]  = 1
    shape_list[0] = n_models
    shape_list[1] = num_set_indices
    list_fns = np.zeros(shape_list)
    
    for i_model, model in enumerate(list_models):
        list_fns[i_model] = [model.compute_fn_no_pseudocount(n, indices) for indices in set_indices]
    return list_fns


'''
Functions that compute 3- and 4-point marginals and connected correlations for a list of Correlations objects,
at a set of num_set_indices random site positions, over all letters at those positions.
'''
def produce_f3s_and_CM3s_1free(list_models, num_set_indices=1):
    n_models = len(list_models)
    L = list_models[0].L
    q = list_models[0].q
    set_indices = np.sort(np.random.randint(L, size=(num_set_indices, 2)))
    
    list_f3s = np.zeros((n_models, num_set_indices, 1, q, 1, q, L, q))
    list_CM3s = np.zeros((n_models, num_set_indices, 1, q, 1, q, L, q))
    
    for i_model, model in enumerate(list_models):
        list_f3s[i_model] = [model.compute_f3_no_pseudocount_1free(indices) for indices in set_indices]
        list_CM3s[i_model] = [model.compute_CM3(indices) for indices in set_indices]
        
    return [list_f3s, list_CM3s]


def produce_f4s_and_CM4s(list_models, num_set_indices=1):
    n_models = len(list_models)
    L = list_models[0].L
    q = list_models[0].q
    set_indices = np.sort(np.random.randint(L, size=(num_set_indices, 4)))

    list_f4s = np.zeros((n_models, num_set_indices, 1, q, 1, q, 1, q, 1, q))
    list_CM4s = np.zeros((n_models, num_set_indices, 1, q, 1, q, 1, q, 1, q))
    
    for i_model, model in enumerate(list_models):
        list_f4s[i_model] = [model.compute_f4_no_pseudocount(indices) for indices in set_indices]
        list_CM4s[i_model] = [model.compute_CM4(indices) for indices in set_indices] 
    return [list_f4s, list_CM4s]


def compute_r_ns(list_tensors, ref_model_index = 0, n_largest = 20):
    '''
    Compute the r_20 measures of a given list of tensors, with respect to a reference tensor.
    Parameters:
    - list_tensors: (list of np.arrays) the tensors to be compared, include the reference.
    - ref_model_index: (int) index of the reference tensor within the list_tensors
    - n_largest: (int) indicates the number of the largest components used to compute the Pearson r's.
        The default value is 20, corresponding to the r_20 metric.
    '''
    n_tensors = len(list_tensors)-1
    num_set_indices = list_tensors[0].shape[0]
    r_ns = np.zeros((num_set_indices, n_tensors))
    for i_indices in range(num_set_indices):
        ind = np.unravel_index(np.argsort(list_tensors[ref_model_index][i_indices], axis=None), list_tensors[ref_model_index][i_indices].shape)
        largest_components_ref = list_tensors[ref_model_index][i_indices][ind][-n_largest:]
        shift = 0
        for i_tensors in range(n_tensors):
            if i_tensors == ref_model_index:
                shift = 1 
            components = list_tensors[i_tensors+shift][i_indices][ind][-n_largest:]
            r_ns[i_indices, i_tensors] = pearsonr(largest_components_ref, components)[0]
    return  np.array([r_20s for r_20s in r_ns if not np.sum(np.isnan(r_20s)) ])


#Function to quickly compute and visualise average and std_dev of the entries of a tensor
def compute_final_rs(tensor, print_Q = True):
    rs_mean = np.mean(tensor, axis=0)
    rs_std_dev = np.std(tensor, axis=0)
    if print_Q:
        print(f"Result from {len(tensor)} samples (out of {len(tensor)}):")
        print(np.array([rs_mean, (rs_mean - rs_mean[0])*100, rs_std_dev]))
    return [rs_mean, rs_std_dev]


def compute_r20s(list_models, order, out_file_name,
                N_samples = 3000, ref_model_index = 0,
                comparisons_location = './results/correlations/_comparisons'):
    '''
    Computes the marginals of order 'order' of a list of Correlations objects over 'N_samples' random sets 
    of site positions of length 'order'.
    The r_20 measures of these marginals with respect to the reference model is then computed and saved to file.
    The r_20 is computed for each set of site positions, and then averaged over the N_samples.
    Parameters:
    - list_models: (list of Correlations) the MSAs to be compared, including the reference.
    - order: (int) order of the marginals to be computed. Implemented for 3 <= order <= 6.
    - out_file_name: (string) name of the output file containing the r_20s.
    - N_samples: (int) number of sets of site positions (each with length equal to 'order') from which the r_20's are estimated.
    - ref_model_index: (int) index of the reference model within list_models.
    - comparisons_location: (string) location of the folder where the files are written.
    '''
    if order > 6:
        raise Exception("Only order <= 6 is allowed.")   
    order_function_map = {
        3 : produce_f3s,
        4 : produce_f4s,
        5 : produce_f5s,
        6 : produce_f6s
    } 
    all_r_20_fns = np.zeros((N_samples, len(list_models)-1))
    for i_indices in tqdm.tqdm(range(N_samples)):     
        list_fs = order_function_map[order](list_models, num_set_indices=1)
        try:
            all_r_20_fns[i_indices] = compute_r_ns(list_fs, ref_model_index = ref_model_index)[0]
        except:
            continue
    np.save(comparisons_location + os.sep + out_file_name + "_all_r20s.npy", all_r_20_fns) 
    final_rs = compute_final_rs(all_r_20_fns, print_Q=False)
    np.save(comparisons_location + os.sep + out_file_name + ".npy", final_rs) 
    print(f"Average r_20s:\n{final_rs[0]}")
    print(f"Std_dev:\n{final_rs[1]}")



