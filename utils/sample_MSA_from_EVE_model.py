import os
import sys
from functools import partial
from contextlib import redirect_stdout
from timeit import default_timer as timer
import tqdm
import json
import math
import torch

from correlations import *
from utils.compare_marginals import *
from utils.various_tools import *
from EVE import VAE_model


#Auxiliary functions to find site-wise thresholds given a decoded sample from EVE:
def _compute_weight_gap(seq,list_seq, theta):
    number_non_empty_positions = torch.dot(seq,seq)
    if number_non_empty_positions>0.:
        denom = torch.matmul(list_seq,seq) / number_non_empty_positions 
        denom = torch.sum(denom > 1 - theta) 
        return 1/denom
    else:
        return 0.0 


def _compute_1hot(list_recon_x_subsample,thresholds):
    list_recon_x_subsample_max_indices, list_recon_x_subsample_max_values = multisample_from_softmax_MSA(list_recon_x_subsample)
    list_recon_x_subsample_max_1 = torch.greater(list_recon_x_subsample_max_values,thresholds) + 0.
    recon_x_subsample_1hot = torch.zeros_like(list_recon_x_subsample)
    return recon_x_subsample_1hot.scatter_(2, list_recon_x_subsample_max_indices.unsqueeze(-1), list_recon_x_subsample_max_1.unsqueeze(-1))     
    
    
def _compute_weights(recon_x_subsample_1hot,theta = 0.2):
    list_seq = recon_x_subsample_1hot.reshape((recon_x_subsample_1hot.shape[0], 
                                           recon_x_subsample_1hot.shape[1] * recon_x_subsample_1hot.shape[2]))
    _compute_weight_gap_partial = partial(_compute_weight_gap, list_seq = list_seq, theta = theta)
    recon_x_subsample_weights = torch.tensor(list(map(_compute_weight_gap_partial,list_seq)),dtype=recon_x_subsample_1hot.dtype, device=recon_x_subsample_1hot.device)    
    return recon_x_subsample_weights
    
    
def _compute_f1_gaps(recon_x_subsample_1hot,recon_x_subsample_weights, pseudocount = 0.5):
    recon_x_subsample_1hot_filled = torch.sum(recon_x_subsample_1hot,dim=2)
    Neff = torch.sum(recon_x_subsample_weights)
    M = Neff
    f1_no_pseudocount = 1-torch.matmul(recon_x_subsample_weights,recon_x_subsample_1hot_filled)/Neff
    return (pseudocount / ((recon_x_subsample_1hot.shape[2]+1)*M)) + (1. - pseudocount/M) * f1_no_pseudocount    


def _fun(f1_gap_i_site,PABP_f1_gap_i_site):
    return f1_gap_i_site - PABP_f1_gap_i_site


def compute_thresholds(list_recon_x_subsample, ref_f1_gap, 
                       _TOL_per_site = 0.01,
                       n_rep_max = 20,
                       safety_unbalance = 1,
                       avg = False,
                       debug = True):   
    '''
    Compute site-wise thresholds to make the gap 1-site frequencies of a decoded sample from EVE close 
    (within a tolerance and finite-sample error) to the gap (weighted) frequencies of a reference sample.
    The thresholds are fitted with a simple bisection algorithm, which is particularly solid against finite-sample resolution errors.
    Parameters:
    - list_recon_x_subsample: (torch.tensor) the decoded sample from EVE (typically obtained by sampling from the latent space prior).
    - ref_f1_gap: (torch.tensor) reference 1-site gap frequencies.
    - _TOL_per_site: (float) tolerance that establishes when the thresholds fit at each site is complete.
    - n_rep_max: (int) maximum number of steps for the thresholds fitting algorithm.
    - safety_unbalance: (float) additional parameter that slows down the algorithm while making it more stable against finite-sample resolution errors.
        if set to 1, the fitting is faster.
    - avg: (bool) if n_rep_max iterations are hit and not all thresholds are fitted within the tolerance, 
        avg == True fills the unfitted thresholds with the average of the fitted thresholds,
        avg == False fills the unfitted thresholds with zeros (at those sites there will be no gap).
    - debug: (bool) prints the log of the fitting procedure.
    '''
    if debug:
        list_Neff_0 = []
        list_f1_gaps_0 = torch.tensor([],device=list_recon_x_subsample.device)
        list_f1_gaps_1 = torch.tensor([],device=list_recon_x_subsample.device)
    thresholds = torch.zeros(list_recon_x_subsample.shape[1]).to(list_recon_x_subsample.device) 
    thresholds_0 = 0. * torch.ones(list_recon_x_subsample.shape[1]).to(list_recon_x_subsample.device)
    thresholds_1 = 1.1 * torch.ones(list_recon_x_subsample.shape[1]).to(list_recon_x_subsample.device)
    n_rep = 1

    while n_rep < n_rep_max+1: 
        subsample_1hot_0 = _compute_1hot(list_recon_x_subsample,thresholds_0)
        subsample_1hot_1 = _compute_1hot(list_recon_x_subsample,thresholds_1)
        w_0 = _compute_weights(subsample_1hot_0)
        w_1 = _compute_weights(subsample_1hot_1)
        f1_gaps_0 = _compute_f1_gaps(subsample_1hot_0,w_0)
        f1_gaps_1 = _compute_f1_gaps(subsample_1hot_1,w_1)

        if debug:
            list_Neff_0.append(torch.sum(w_0).item())
            list_f1_gaps_0 = torch.cat((list_f1_gaps_0,f1_gaps_0),0)
            list_f1_gaps_1 = torch.cat((list_f1_gaps_1,f1_gaps_1),0)

        thresholds_mid = (thresholds_0+thresholds_1)/2
        subsample_1hot_mid = _compute_1hot(list_recon_x_subsample,thresholds_mid)
        w_mid = _compute_weights(subsample_1hot_mid)
        f1_gaps_mid = _compute_f1_gaps(subsample_1hot_mid,w_mid)

        for i_site in range(list_recon_x_subsample.shape[1]): 
            if thresholds[i_site].item() > 1e-8:
                continue
            fun_mid = _fun(f1_gaps_mid[i_site],ref_f1_gap[i_site])                         
            if torch.abs(fun_mid).item() < _TOL_per_site: 
                thresholds[i_site] = torch.clone(thresholds_mid[i_site])
                continue
            fun_0 = _fun(f1_gaps_0[i_site],ref_f1_gap[i_site]) 
            if torch.sign(fun_mid).item() == torch.sign(fun_0).item():
                thresholds_0[i_site] = torch.clone((safety_unbalance*thresholds_0[i_site]+thresholds_1[i_site])
                                                   /(safety_unbalance+1))
            else: 
                thresholds_1[i_site] = torch.clone((thresholds_0[i_site]+safety_unbalance*thresholds_1[i_site])
                                                   /(safety_unbalance+1))
        if debug:       
            print(f"n_rep = {n_rep}, Filled {torch.count_nonzero(thresholds)} / {torch.numel(thresholds)} components.")
            print(thresholds)            
        if torch.sum(torch.greater(thresholds,1e-8)).item() == thresholds.shape[0]:
            print(f"Thresholds correctly fitted in {n_rep} reps.")
            break      
        n_rep +=1

        if n_rep == n_rep_max+1:
            print("Not enough reps to determine the thresholds to the specified level of accuracy.")
            if avg:
                print("Filling the undetermined thresholds with the average of the others.")
                avg_entry = torch.sum(thresholds)/torch.sum(torch.greater(thresholds,0)).item()
                thresholds[thresholds<1e-4] = avg_entry
            else:
                thresholds[thresholds<1e-4] = 0    
    return thresholds


def multisample_from_softmax_MSA(list_recon_x):
    """
    Sample from a list of sequences, where at each token a softmax probability is given 
    over the alphabet letters.
    res.shape: (torch.tensor) shaped (N, L, q).
    Return torch.tensor shaped (N, L).
    """
    N = list_recon_x.shape[0]
    size = list_recon_x.shape[1]
    rand_values = torch.rand((N, size, 1), device=list_recon_x.device)
    cumprobs = list_recon_x.cumsum(dim=2)
    sampled = torch.searchsorted(cumprobs, rand_values).squeeze(2)
    sampled_probs = torch.gather(list_recon_x, -1, sampled.unsqueeze(-1) ).squeeze(-1)
    return sampled, sampled_probs


def select_and_write_MSA(list_recon_x,
                         thresholds,
                         file_name,
                         prefix,
                         focus_line = None
                        ):
    '''
    Given site-wise thresholds and a decoded sample from EVE, select and write sequences to a file.
    Parameters:
    - list_recon_x: (torch.tensor) decoded sample from EVE.
    - thresholds: (torch.tensor) site-wise thresholds to identify gap vs AA at each site.
    - file_name: (string) location when the generated MSA is written to.
    - prefix: (string) prefix in the label of each generated sequence. It is then followed by the number of the sequence within the MSA.
    - focus_line: (string) label + '\n' + sequence for the focus sequence to be prepended to the generated MSA.
    '''
    list_recon_x_sampled, list_recon_x_sampled_probs = multisample_from_softmax_MSA(list_recon_x)
    list_recon_x_sampled_indices = torch.greater(list_recon_x_sampled_probs,thresholds)*(list_recon_x_sampled+1)
    list_recon_x_sampled_indices_np = torch.Tensor.numpy(list_recon_x_sampled_indices.cpu())
    list_recon_x_letters = np.array(list(alphabetg))[list_recon_x_sampled_indices_np]
    list_recon_x_sequences = list_recon_x_letters.view('U' + str(list_recon_x_letters.shape[1])).ravel()
    with open(file_name, 'w') as file:
        if focus_line is not None:
            file.write(focus_line + '\n')
        for i_seq, sequence in enumerate(list_recon_x_sequences):
            file.write(prefix+str(i_seq)+'\n')
            file.write(sequence+'\n')   


def load_EVE_model(in_weights_file_name,
                    in_model_name,
                    in_MSA_name,
                    in_VAE_checkpoint_location,
                    model_parameters_location='./EVE/default_model_params.json',
                    MSA_data_folder='./data/MSA',
                    MSA_weights_folder='./data/weights',
                    theta = 0.2
                    ):            
    '''
    Load and return EVE model.
    Parameters:
    - in_weights_file_name: (string) name of the npy file containing the weights of the training MSA used for the model.
    - in_model_name: (string) name of the EVE model to be loaded.
    - in_MSA_name: (string) name of the file containing the training MSA used for the model. Include extension.
    - in_VAE_checkpoint_location: (string) full location of the EVE checkpoint to be loaded.
    - model_parameters_location: (string) location of the file determining the model hyperparameters.
    - MSA_data_folder: (string) folder location of MSAs.
    - MSA_weights_folder: (string) folder location of weights files.
    - theta: (float) theta parameter used to compute the weights of the training MSA.
    '''
    msa_location = MSA_data_folder + os.sep + in_MSA_name  
    print("-----------------------------------------------")      
    print("Loading data.")
    data = data_utils.MSA_processing(
        MSA_location=msa_location,
        theta=theta,
        use_weights=True,
        weights_location=MSA_weights_folder + os.sep + in_weights_file_name + ".npy"
    )   
    print("-----------------------------------------------")    
    print("Loading model.")
    print("Model name: " + str(in_model_name))
    model_params = json.load(open(model_parameters_location))
    model = VAE_model.VAE_model(
                    model_name=in_model_name,
                    data=data,
                    encoder_parameters=model_params["encoder_parameters"],
                    decoder_parameters=model_params["decoder_parameters"],
                    random_seed=42
    )
    model = model.to(model.device)
    try:
        checkpoint = torch.load(in_VAE_checkpoint_location)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Initialized VAE with checkpoint " +  in_VAE_checkpoint_location)
    except:
        print("Unable to locate VAE model checkpoint")
        sys.exit(0)
    return model


def generate_MSA_and_Correlations_from_EVE_model(ref_weights_file_name,
                            ref_Correlations_name,
                            ref_MSA_name,
                            in_VAE_checkpoint_location,
                            out_Correlations_name,
                            N_samples = 10000,
                            threshold = None,
                            N_subsample = 5000,
                            MSA_data_folder='./data/MSA',
                            MSA_weights_folder='./data/weights',
                            correlations_folder = './results/correlations',
                            model_parameters_location='./EVE/default_model_params.json',
                            n_rep_max = 25,
                            _TOL_per_site = 0.01,
                            avg_thresholds = False,
                            debug = True,
                            theta = 0.2
                           ):
    '''
    Sample an MSA from an EVE checkpoint (using variable site-wise thresholds or fixed site-independent thresholds), 
    write it to file and compute its marginals and correlations. Return and write to file the resulting Correlations object.
    Parameters:
    - ref_weights_file_name: (string) name of the npy file containing the weights of the training MSA used for the model.
    - ref_Correlations_name: (string) name of the Correlations object containing marginals and correlations of the training set.
    - ref_MSA_name: (string) name of the file containing the training MSA used for the model. Include extension.
    - in_VAE_checkpoint_location: (string) full location of the EVE checkpoint to be loaded.
    - out_Correlations_name: (string) name of the output Correlations object.
    - N_samples: (int) number of sequences to be sampled.
    - threshold: (float) site-independent threshold to determine whether at a each site of each decoded sequence there is a gap or an AA.
        If not provided, variable site-wise thresholds are fitted to reproduce the 1-site gap (weighted) frequencies in the training MSA.
    - N_subsample: (int) size of the subsample of the generated sequences used to fit the thresholds. If threshold is not None, this parameter is ignored.
    - MSA_data_folder: (string) folder location of MSAs.
    - MSA_weights_folder: (string) folder location of weights files.
    - correlations_folder: (string) folder location of Correlations files.
    - model_parameters_location: (string) location of the file determining the model hyperparameters.
    - n_rep_max: (int) maximum number of steps for the thresholds fitting algorithm. If threshold is not None, this parameter is ignored.
    - _TOL_per_site: (float) tolerance that establishes when the thresholds fit at each site is complete. If threshold is not None, this parameter is ignored.
    - avg_thresholds: (bool) if n_rep_max iterations are hit and not all thresholds are fitted within the tolerance, 
        avg_thresholds == True fills the unfitted thresholds with the average of the fitted thresholds,
        avg_thresholds == False fills the unfitted thresholds with zeros (at those sites there will be no gap).
        If threshold is not None, this parameter is ignored.
    - debug: (bool) prints the log of the thresholds fitting procedure. If threshold is not None, this parameter is ignored.
    - theta: (float) theta parameter used to compute the weights of the training MSA.
    '''
    start_time = timer()
    if not debug:
        standard_out_old = sys.stdout
        f = open('/dev/null', 'w')
        sys.stdout = f
    ref_Correlations = load_instance_from_file(correlations_folder + os.sep + ref_Correlations_name 
                                       + os.sep + ref_Correlations_name + ".Correlations")

    model = load_EVE_model(in_weights_file_name = ref_weights_file_name,
                            in_model_name = ref_Correlations_name,
                            in_MSA_name = ref_MSA_name,
                            in_VAE_checkpoint_location = in_VAE_checkpoint_location,
                            model_parameters_location = model_parameters_location,
                            MSA_data_folder = MSA_data_folder,
                            MSA_weights_folder = MSA_weights_folder,
                            theta = theta
                            )
    print("-----------------------------------------------")     
    print(f"Generating {N_samples} sequences.")
    list_z_sampled = torch.zeros((N_samples,model.z_dim)).to(model.device)
    for ii in range(N_samples):
        list_z_sampled[ii] = model.sample_latent_prior()
    with torch.no_grad():
        list_recon_x = torch.nn.functional.softmax(model.decoder(list_z_sampled), dim=2)
    
    if threshold is None:
        ref_f1_gap = torch.tensor(ref_Correlations.f1[:,0]).to(model.device)
        print(f"Searching for the appropriate gap thresholds.") 
        list_recon_x_subsample = list_recon_x[::N_samples//N_subsample][:N_subsample]
        thresholds = compute_thresholds(list_recon_x_subsample,                    
                                        ref_f1_gap = ref_f1_gap, 
                                        _TOL_per_site = _TOL_per_site, 
                                        n_rep_max = n_rep_max, 
                                        debug = debug, 
                                        avg = avg_thresholds
                                        )
        subsample_1hot_end = _compute_1hot(list_recon_x,thresholds)
        w_end = _compute_weights(subsample_1hot_end)
        f1_gaps_end = _compute_f1_gaps(subsample_1hot_end,w_end)    
        avg_deviation = torch.sum(torch.abs(f1_gaps_end - ref_f1_gap))/ref_f1_gap.shape[0]
        max_deviation = torch.max(f1_gaps_end - ref_f1_gap)
        print(f"Average deviation per site: {avg_deviation.item():.4f}, \nMax deviation per site: {max_deviation.item():.4f}")
    else:
        if threshold == -1:
            print(f"Obtaining gap threshold as {len(    alphabet)}-th percentile from decoded MSA.") #print(f"Obtaining gap threshold as {len(ref_Correlations.alphabet)}-th percentile from decoded MSA.")
            index = int( list_recon_x.shape[0]*list_recon_x.shape[1])
            threshold = torch.sort(list_recon_x.view(-1), descending=True).values[index].cpu().item()
        print(f"Using gap threshold {threshold}.")
        thresholds = threshold*torch.ones(list_recon_x.shape[1]).to(model.device) 
    print("Generation completed.")
    print("-----------------------------------------------")     
    ref_focus_key, ref_focus_seq = next(enumerate(ref_Correlations.label_to_seq.items()))[1]
    prefix = ">" + out_Correlations_name + "_VAE_generated/Seq_"
    focus_line = ref_focus_key + '\n' + "".join(ref_focus_seq)
    out_MSA_file_name = MSA_data_folder + os.sep + out_Correlations_name + '.a2m'

    print("Writing generated MSA to file at " + out_MSA_file_name)
    select_and_write_MSA(list_recon_x,thresholds,
                         file_name = out_MSA_file_name,
                         prefix = prefix,
                         focus_line = focus_line
                        )
    out_correlations_subfolder = correlations_folder + os.sep + out_Correlations_name + os.sep
    shell_run('mkdir -p ' + out_correlations_subfolder)
    file_name_label_to_seq_out = out_correlations_subfolder + out_Correlations_name + "_label_to_seq.npy"
    file_name_weights_out = MSA_weights_folder + os.sep + out_Correlations_name + ".npy"
    out_Correlations = Correlations(MSA_location = out_MSA_file_name, MFA = False,
                            file_name_label_to_seq_out = file_name_label_to_seq_out,
                            file_name_weights_out = file_name_weights_out,
                            advanced_preprocess_MSA = False
                        )
    print(f"Computing weights of the generated MSA.")
    timed(out_Correlations.compute_weights)()
    print(f"N_eff = {out_Correlations.Neff:.2f}")
    print(f"Computing correlation metrics of the generated MSA.")    
    out_Correlations.compute_all()
    save_instance_to_file(out_correlations_subfolder + out_Correlations_name + '.Correlations', out_Correlations)
    end_time = timer()
    print(f'Completed. Total elapsed time:' + to_minutes(end_time, start_time) + 'min')    
    if not debug:
        sys.stdout = standard_out_old
    return out_Correlations 


def generate_MSA_and_Correlations_from_EVE_models(ref_weights_file_name,
                                                    ref_Correlations_name,
                                                    ref_MSA_name,
                                                    in_VAE_checkpoint_location,
                                                    out_Correlations_name,
                                                    initial_epoch = 10*1000,
                                                    final_epoch = 1400*1000,
                                                    step = 10*1000,
                                                    N_samples = 10000,
                                                    threshold = None,
                                                    N_subsample = 5000,
                                                    MSA_data_folder='./data/MSA',
                                                    MSA_weights_folder='./data/weights',
                                                    correlations_folder = './results/correlations',
                                                    model_parameters_location='./EVE/default_model_params.json',
                                                    n_rep_max = 25,
                                                    _TOL_per_site = 0.01,
                                                    avg_thresholds = False,
                                                    debug = True,
                                                    theta = 0.2,
                                                    additional_log_tag = ''
                                                    ):
    '''
    Generates MSAs from different checkpoints during EVE training, and computes their marginals and correlations.
    The checkpoints are assumed to be all in the same folder and have the same name, except for the last digits 
    which encode the training steps of each checkpoint. 
    Parameters:
    - ref_weights_file_name: (string) name of the npy file containing the weights of the training MSA used for the model.
    - ref_Correlations_name: (string) name of the Correlations object containing marginals and correlations of the training set.
    - ref_MSA_name: (string) name of the file containing the training MSA used for the model. Include extension.
    - in_VAE_checkpoint_location: (string) full location of the EVE checkpoints to be loaded, modulo the final "_step_" + str(n_steps).
    - out_Correlations_name: (string) name of the output Correlations objects, to which '_' + str(n_steps//1000) is appended in each case.
    - initial_epoch: (integer) training steps of the first checkpoint analysed.
    - final_epoch: (integer) training steps of the last checkpoint analysed.
    - step: (integer) stride in training steps between two analysed checkpoints.
    - N_samples: (int) number of sequences to be sampled.
    - threshold: (float) site-independent threshold to determine whether at a each site of each decoded sequence there is a gap or an AA.
        If not provided, variable site-wise thresholds are fitted to reproduce the 1-site gap (weighted) frequencies in the training MSA.
    - N_subsample: (int) size of the subsample of the generated sequences used to fit the thresholds. If threshold is not None, this parameter is ignored.
    - MSA_data_folder: (string) folder location of MSAs.
    - MSA_weights_folder: (string) folder location of weights files.
    - correlations_folder: (string) folder location of Correlations files.
    - model_parameters_location: (string) location of the file determining the model hyperparameters.
    - n_rep_max: (int) maximum number of steps for the thresholds fitting algorithm. If threshold is not None, this parameter is ignored.
    - _TOL_per_site: (float) tolerance that establishes when the thresholds fit at each site is complete. If threshold is not None, this parameter is ignored.
    - avg_thresholds: (bool) if n_rep_max iterations are hit and not all thresholds are fitted within the tolerance, 
        avg_thresholds == True fills the unfitted thresholds with the average of the fitted thresholds,
        avg_thresholds == False fills the unfitted thresholds with zeros (at those sites there will be no gap).
        If threshold is not None, this parameter is ignored.
    - debug: (bool) prints the log of the thresholds fitting procedure. If threshold is not None, this parameter is ignored.
    - theta: (float) theta parameter used to compute the weights of the training MSA.
    - additional_log_tag: (string) additional label to be appended at the end of the log file name.
    '''
    with open('./logs/generation_' + ref_Correlations_name + "_to_" + out_Correlations_name + "_" + additional_log_tag + '.txt', 'w') as f:
        with redirect_stdout(f):
            for i_model in tqdm.tqdm(range(int((final_epoch-initial_epoch)/step)+1), 'Looping through models'):
                n_steps = initial_epoch + step*i_model
                VAE_checkpoint_location = in_VAE_checkpoint_location + "_step_" + str(n_steps)
                out_model_file_name = out_Correlations_name + '_' + str(n_steps//1000)

                generate_MSA_and_Correlations_from_EVE_model(ref_weights_file_name = ref_weights_file_name,
                                                ref_Correlations_name = ref_Correlations_name,
                                                ref_MSA_name = ref_MSA_name,
                                                in_VAE_checkpoint_location = VAE_checkpoint_location,
                                                out_Correlations_name = out_model_file_name,
                                                N_samples = N_samples,
                                                threshold = threshold,
                                                N_subsample = N_subsample,
                                                MSA_data_folder = MSA_data_folder,
                                                MSA_weights_folder = MSA_weights_folder,
                                                correlations_folder = correlations_folder,
                                                model_parameters_location= model_parameters_location,
                                                n_rep_max = n_rep_max,
                                                _TOL_per_site = _TOL_per_site,
                                                avg_thresholds = avg_thresholds,
                                                debug = debug,
                                                theta = theta
                                            )


def compute_performance_EVE_models_vs_ref(name_performance,
                                        ref_Correlations_name,
                                        EVE_Correlations_name,
                                        labels = ['f1', 'f2', 'CM2', 'MI'], 
                                        input_results_location = None, 
                                        initial_epoch = 10*1000,
                                        final_epoch = 1400*1000,
                                        step = 10*1000,
                                        correlations_folder = './results/correlations'
                                       ):
    '''
    Compute and write to file the performance of an EVE model over training in reproducing the marginals and correlations 
    of a reference Correlations object (typically, encoding marginals and correlations of the training or test set.)
    Parameters:
    - name_performance: (string) name of the performance file containing the Pearson r's of marginals and correlations 
        between the analysed EVE models and the reference Correlations object.
    - ref_Correlations_name: (string) name of the Correlations object containing marginals and correlations of the training set. 
    - EVE_Correlations_name: (string) name of the input EVE Correlations objects, modulo the '_' + str(n_steps//1000) final part.
    - labels: (list of strings) set of marginals and correlations to be compared between the reference and the EVE Correlations objects.
    - input_results_location: (string) location of file where current results are to be appended to. 
        Useful if additional EVE training checkpoints are being analysed.
    - initial_epoch: (integer) training steps of the first EVE Correlations object to be analysed.
    - final_epoch: (integer) training steps of the last EVE Correlations object to be analysed.
    - step: (integer) stride in training steps between two analysed EVE Correlations objects.
    - correlations_folder: (string) folder location of Correlations files.
    '''
    if input_results_location is None:
        results = {}
    else: 
        results = np_load_dict(input_results_location)
    results["index"] = labels
    file_name_out = correlations_folder + os.sep + "_performances" + os.sep + name_performance + ".npy"
    ref_Correlations = load_instance_from_file(correlations_folder + os.sep + ref_Correlations_name 
                                       + os.sep + ref_Correlations_name + ".Correlations")

    with open('./logs/performance_' + name_performance + '.txt', 'w') as f:
        with redirect_stdout(f):
            for i_model in tqdm.tqdm(range(int((final_epoch-initial_epoch)/step)+1), 'Looping through models'):
                n_steps = initial_epoch + step*i_model
                results_single_model = np.array([]).reshape(0,len(labels))
                EVE_model_file_name = (correlations_folder + os.sep + EVE_Correlations_name + '_' + str(n_steps//1000) 
                                       + os.sep + EVE_Correlations_name + '_' + str(n_steps//1000) + ".Correlations")
                VAE_Correlations = load_instance_from_file(EVE_model_file_name)

                for label in results["index"]:
                    vec_results = eval(f"compare_tensors(VAE_Correlations.{label}, ref_Correlations.{label}, plot = False, return_Q= True)")
                    results_single_model = np.vstack((results_single_model, vec_results))
                results[str(n_steps//1000)] = [VAE_Correlations.Neff, results_single_model]  
    np.save(file_name_out, results) 
    print("Performance file successfully saved to " + file_name_out)               