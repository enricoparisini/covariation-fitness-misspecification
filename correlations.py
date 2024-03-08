from numba import jit
import numpy as np
from functools import partial
from utils import data_utils


'''
Auxiliary functions to speed up computations with numba in the class Correlations
'''
@jit(nopython=True)
def _compute_weight(seq,list_seq,theta):
    number_non_empty_positions = np.dot(seq,seq)
    if number_non_empty_positions>0:
        denom = np.dot(list_seq,seq) / number_non_empty_positions 
        #Number of letters common to both seq and each sequence in list_seq / number_non_empty_positions
        # = % of letters in each sequence common to seq
        denom = np.sum(denom > 1 - theta) 
        return 1/denom
    else:
        return 0.0 #return 0 weight if sequence is fully empty

def _compute_weights(label_to_OHE,theta):
    list_seq = label_to_OHE.reshape(
        (label_to_OHE.shape[0], label_to_OHE.shape[1] * label_to_OHE.shape[2])
        )
    compute_weight_partial = partial(_compute_weight, list_seq = list_seq, theta = theta)  
    return np.array(list(map(compute_weight_partial,list_seq)))

@jit(nopython=True)
def _compute_f1(f1_no_pseudocount, pseudocount, M):
    q = f1_no_pseudocount.shape[1]
    return  (pseudocount / (q*M)) + (1. - pseudocount/M) * f1_no_pseudocount 

@jit(nopython=True)
def _compute_f2(f1_no_pseudocount,f2_no_pseudocount,pseudocount, M):
    L, q = f1_no_pseudocount.shape
    f2 =  pseudocount / float(q*q*M) + (1. - pseudocount/M) * f2_no_pseudocount
    id_matrix = np.identity(q)
    for i in range(L):
        f2[i, :, i, :] = (
        (1. - pseudocount/M) * f2_no_pseudocount[i, :, i, :] 
        + (pseudocount / float(q*M)) * id_matrix
        )
    return f2
    
@jit(nopython=True)
def _compute_covariance_matrix(f1, f2):  
    L, q = f1.shape
    CM = np.zeros(( L * (q - 1), L * (q - 1) ))    
    for i in range(L):
        for j in range(L):
            for A in range(q - 1):
                for B in range(q - 1):
                    CM[i * (q - 1) + A, j * (q - 1) + B] = f2[i, A, j, B] - f1[i, A] * f1[j, B]
    return CM

@jit(nopython=True)
def _compute_couplings_e(CM, L, q):
    E_flat = - np.linalg.inv(CM)   
    E = np.zeros((L, q, L, q))
    for i in range(L):
        for j in range(L):
            for A in range(q - 1):
                for B in range(q - 1):
                    E[i, A, j, B] = E_flat[i * (q - 1) + A, j * (q - 1) + B]
    return E

@jit(nopython=True)
def _compute_H_tildes(f_i, f_j, expE_ij):
    _EPSILON = 1e-5
    delta_H = 1.0
    q = f_i.shape[0]
    H_i = np.full((1, q), 1 / float(q))
    H_j = np.full((1, q), 1 / float(q))
    while delta_H > _EPSILON:
        H_i_updated = f_i / np.dot(H_j, expE_ij.T)
        H_i_updated /= H_i_updated.sum()
        H_j_updated = f_j / np.dot(H_i, expE_ij)
        H_j_updated /= H_j_updated.sum()

        delta_H = max(
            np.absolute(H_i_updated - H_i).max(),
            np.absolute(H_j_updated - H_j).max()
        )
        H_i = H_i_updated
        H_j = H_j_updated
    return H_i, H_j    

@jit(nopython=True)
def _compute_direct_information(E, f1):
    L, q = f1.shape
    DI = np.zeros((L, L))   
    for i in range(L):
        for j in range(i + 1, L):
            expE_ij = np.exp(E[i, :, j,:])
            H_i, H_j = _compute_H_tildes(f1[i],f1[j],expE_ij)
            P_dir_ij = expE_ij * np.dot(H_i.T, H_j)
            P_dir_ij = P_dir_ij / P_dir_ij.sum()
            den = np.dot(f1[i].reshape((1, q)).T, f1[j].reshape((1, q)))         
            DI[i, j] = DI[j, i] = np.sum(P_dir_ij * np.log((P_dir_ij + 1.0e-30) / (den + 1.0e-30)))
    return DI  

@jit(nopython=True)
def _compute_mutual_information(f1, f2):    
    L, q = f1.shape
    MI = np.zeros((L,L))
    for i in range(L):
        for j in range(L):
            den = np.dot( f1[i].reshape((1, q)).T, f1[j].reshape((1, q)))
            MI[i, j] = MI[j, i] = np.sum(f2[i, :, j, :] * np.log((f2[i, :, j, :] + 1.0e-30) / (den + 1.0e-30)))
    return MI



class Correlations(data_utils.basic_MSA_processing):  
    #Class containing the tools to compute and manipulate marginals and correlations
    def __init__(self, MSA_location = None, label_to_seq=None, weights=None,
                file_name_weights_in = None, 
                theta=0.2, pseudocount=0.5, MFA = False,
                preprocess_MSA=True,
                advanced_preprocess_MSA = True,
                threshold_sequence_frac_gaps=0.5,
                threshold_focus_cols_frac_gaps=0.3,
                keep_cols = None,
                remove_sequences_with_indeterminate_AA_in_focus_cols=True,
                file_name_label_to_seq_out = None,
                file_name_weights_out = None):
        '''
        Parameters:
        - MSA_location: (string) location of the input MSA.
        - label_to_seq: (dict) MSA in dictionary label_to_seq format to initialise alignment if no MSA_location is provided.
        - file_name_weights_in: (string) location of the weights of the alignment. 
        - weights: (np array) weights of each sequence in the alignment based on similarity.
            If neither weights nor file_name_weights_in are provided, a unitary weight is assigned to each sequence.
        - theta: (float) sequence weighting hyperparameter.
        - pseudocount: (float) parameter estimating the uncertainty in marginals due to finite-sample effects.
        - MFA: (bool) if True, marginals and correlations are computed with the prescription needed to compute direct information in Mean Field Approximation, 
            following the implementation of EVcouplings at https://github.com/debbiemarkslab/EVcouplings.
            Effectively, this makes the pseudocount considerably larger to make the covariance matrix invertible.
        - preprocess_MSA: (bool) removes columns that are gaps in focus sequence.
        - advanced_preprocess_MSA: (bool) performs pre-processing of MSA to remove short fragments and positions that are not well covered
            exceeding threshold_sequence_frac_gaps and threshold_focus_cols_frac_gaps.
        - threshold_sequence_frac_gaps: (float, between 0 and 1) Threshold value to define fragments
            - sequences with a fraction of gap characters above threshold_sequence_frac_gaps are removed
            - default is set to 0.5 (i.e., fragments with 50% or more gaps are removed)
        - threshold_focus_cols_frac_gaps: (float, between 0 and 1) Threshold value to define focus columns
            - positions with a fraction of gap characters above threshold_focus_cols_pct_gaps will be set to lower case (and not included in the focus_cols)
            - default is set to 0.3 (i.e., focus positions are the ones with 30% of gaps or less, i.e., 70% or more residue occupancy)
        - keep_cols: (list) 2 integers specifying the range of columns to be kept, overriding the gap frequency.
        - remove_sequences_with_indeterminate_AA_in_focus_cols: (bool) Remove all sequences that have indeterminate AA (e.g., B, J, X, Z) at focus positions of the wild type.
        - file_name_label_to_seq_out: (string) the location of the output file containing the dictionary label_to_seq.
        - file_name_weights_out:  (string) the location where the weights are saved after computing them.
        '''
        super().__init__(MSA_location=MSA_location,
                    keep_cols = keep_cols,
                    file_name_label_to_seq_out = file_name_label_to_seq_out,
                    preprocess_MSA=preprocess_MSA,
                    advanced_preprocess_MSA = advanced_preprocess_MSA,
                    threshold_sequence_frac_gaps=threshold_sequence_frac_gaps,
                    threshold_focus_cols_frac_gaps=threshold_focus_cols_frac_gaps,
                    remove_sequences_with_indeterminate_AA_in_focus_cols=remove_sequences_with_indeterminate_AA_in_focus_cols)
        if MSA_location is not None:
            self.msa_to_dict()
        else:
            self.label_to_seq = label_to_seq

        self.theta = theta
        self.pseudocount = pseudocount
        self.MFA = MFA
        self.file_name_weights_out = file_name_weights_out

        self.N = len(self.label_to_seq)
        if file_name_weights_in is not None:
            self.weights = np.load(file_name_weights_in)
            self.Neff = np.sum(self.weights)
        elif weights is not None:
            self.weights = weights
            self.Neff = np.sum(self.weights)
        else:
            self.weights = np.ones(self.N)
            self.Neff = self.N
            
        self.seq_name_focus = next(iter(self.label_to_seq))
        self.seq_focus = self.label_to_seq[self.seq_name_focus][:]
        self.L = len(self.seq_focus)

        self.label_to_OHE = None
        self.f1_no_pseudocount = None
        self.f1 = None
        self.f2_no_pseudocount = None
        self.f2 = None
        self.MI = None
        self.CM = None
        self.E = None
        self.DI = None
        self.CM2 = None
   

    #Accessory function that One-Hot-Encodes the provided MSA
    def _OHE_MSA(self):
        self.label_to_OHE = np.zeros((self.N,self.L,self.q-1))
        for i,seq_name in enumerate(self.label_to_seq.keys()):
            sequence = self.label_to_seq[seq_name]
            for j,letter in enumerate(sequence):
                if letter in self.aa_dict: 
                    self.label_to_OHE[i,j,self.aa_dict[letter]] = 1.0


    #Compute weights of the MSA
    def compute_weights(self):
        if self.label_to_OHE is None:
            self._OHE_MSA()
        self.weights = _compute_weights(self.label_to_OHE,self.theta)
        self.Neff = np.sum(self.weights)
        if self.file_name_weights_out is not None:
            np.save(self.file_name_weights_out, self.weights)


    #Compute 1-site probabilities
    def compute_f1(self):
        if self.MFA:
            M = 1
        else:
            M = self.Neff
        self.f1_no_pseudocount = np.zeros((self.L,self.q))
        for i_seq, sequence in enumerate(self.label_to_seq.values()):
            for i in range(self.L):
                self.f1_no_pseudocount[i,self.aag_dict[sequence[i]]] += self.weights[i_seq]
        self.f1_no_pseudocount = np.divide(self.f1_no_pseudocount, self.Neff)
        self.f1 = _compute_f1(self.f1_no_pseudocount, self.pseudocount, M)
        

    #Compute 2-site probabilities
    def compute_f2(self):
        if self.MFA:
            M = 1
        else:
            M = self.Neff
        if self.f1_no_pseudocount is None:
            self.compute_f1()      
        self.f2_no_pseudocount = np.zeros((self.L,self.q,self.L,self.q))

        for i_seq, sequence in enumerate(self.label_to_seq.values()):
            for i in range(self.L):
                for j in range(i+1, self.L):
                    self.f2_no_pseudocount[i, self.aag_dict[sequence[i]], j, self.aag_dict[sequence[j]]] += self.weights[i_seq]
                    self.f2_no_pseudocount[j, self.aag_dict[sequence[j]], i, self.aag_dict[sequence[i]]] += self.weights[i_seq]
        self.f2_no_pseudocount = np.divide(self.f2_no_pseudocount, self.Neff)   

        for i in range(self.L):
            for A in range(self.q):
                self.f2_no_pseudocount[i, A, i, A] = self.f1_no_pseudocount[i, A]
        self.f2 = _compute_f2(self.f1_no_pseudocount, self.f2_no_pseudocount, self.pseudocount, M)


    #Compute the mutual information based on the 1- and 2-site marginals
    def compute_mutual_information(self):
        if self.f1 is None:
            self.compute_f1()
        if self.f2 is None:
            self.compute_f2() 
        self.MI = _compute_mutual_information(self.f1,self.f2)

    def compute_CM2(self): #all free indices
        self.CM2 =  self.f2_no_pseudocount - np.tensordot(self.f1_no_pseudocount, self.f1_no_pseudocount, axes=0)


    #Compute (flattened) covariance matrix
    def compute_covariance_matrix(self):
        if self.f1 is None:
            self.compute_f1()
        if self.f2 is None:
            self.compute_f2()    
        self.CM = _compute_covariance_matrix(self.f1,self.f2)


    #Compute 2-site couplings of the associated Potts model in Mean Field Approximation
    def compute_couplings_e(self):
        if self.CM is None:
            self.compute_covariance_matrix()
        self.E = _compute_couplings_e(self.CM,self.L, self.q)


    #Compute direct information. 
    #If the 2-site couplings have not been assigned, they are computed in Mean Field Approximation
    def compute_direct_information(self):
        if self.E is None:
            self.compute_couplings_e()
        self.DI = _compute_direct_information(self.E, self.f1)
        

    #Compute all 1- and 2-site marginals and correlations.
    #If MFA is True, it also carries out direct coupling analysis in Mean Field Approximation
    def compute_all(self):
        if self.Neff == self.N:
            self.compute_weights()
        self.compute_f1()
        self.compute_f2()
        self.compute_CM2()
        self.compute_mutual_information()
        if self.MFA:
            self.compute_covariance_matrix()
            self.compute_couplings_e()
            self.compute_direct_information()
    

    #Compute and return the 3-site marginals with two fixed site positions and one free
    def compute_f3_no_pseudocount_1free(self, indices): 
        f3_no_pseudocount = np.zeros((1, self.q, 1, self.q, self.L, self.q))     
        for i_seq, sequence in enumerate(self.label_to_seq.values()):
            for i_3 in range(self.L):
                f3_no_pseudocount[
                    0, self.aag_dict[sequence[indices[0]]],
                    0, self.aag_dict[sequence[indices[1]]], 
                    i_3, self.aag_dict[sequence[i_3]]] += self.weights[i_seq]
        return (f3_no_pseudocount / self.Neff)   


    #Compute and return the 3-site marginals with fixed site positions
    def compute_f3_no_pseudocount(self, indices):
        f3_no_pseudocount = np.zeros((1, self.q, 1, self.q, 1, self.q))
        for i_seq, sequence in enumerate(self.label_to_seq.values()):
            f3_no_pseudocount[
            0, self.aag_dict[sequence[indices[0]]], 
            0, self.aag_dict[sequence[indices[1]]], 
            0, self.aag_dict[sequence[indices[2]]]] += self.weights[i_seq]
        return (f3_no_pseudocount / self.Neff)  


    #Compute and return the 4-site marginals with fixed site positions
    def compute_f4_no_pseudocount(self, indices): 
        f4_no_pseudocount = np.zeros((1, self.q, 1, self.q, 1, self.q, 1, self.q))
        for i_seq, sequence in enumerate(self.label_to_seq.values()):
            f4_no_pseudocount[
            0, self.aag_dict[sequence[indices[0]]], 
            0, self.aag_dict[sequence[indices[1]]], 
            0, self.aag_dict[sequence[indices[2]]], 
            0, self.aag_dict[sequence[indices[3]]]] += self.weights[i_seq]
        return (f4_no_pseudocount / self.Neff)  
    

    #Compute and return the 5-site marginals with fixed site positions
    def compute_f5_no_pseudocount(self, indices): 
        f5_no_pseudocount = np.zeros((1, self.q, 1, self.q, 1, self.q, 1, self.q, 1, self.q))
        for i_seq, sequence in enumerate(self.label_to_seq.values()):
            f5_no_pseudocount[
            0, self.aag_dict[sequence[indices[0]]], 
            0, self.aag_dict[sequence[indices[1]]], 
            0, self.aag_dict[sequence[indices[2]]], 
            0, self.aag_dict[sequence[indices[3]]], 
            0, self.aag_dict[sequence[indices[4]]]] += self.weights[i_seq]
        return (f5_no_pseudocount / self.Neff)  
    

    #Compute and return the 5-site marginals with fixed site positions
    def compute_f6_no_pseudocount(self, indices): 
        f6_no_pseudocount = np.zeros((1, self.q, 1, self.q, 1, self.q, 1, self.q, 1, self.q, 1, self.q))
        for i_seq, sequence in enumerate(self.label_to_seq.values()):
            f6_no_pseudocount[
            0, self.aag_dict[sequence[indices[0]]], 
            0, self.aag_dict[sequence[indices[1]]], 
            0, self.aag_dict[sequence[indices[2]]], 
            0, self.aag_dict[sequence[indices[3]]], 
            0, self.aag_dict[sequence[indices[4]]], 
            0, self.aag_dict[sequence[indices[5]]]] += self.weights[i_seq]
        return (f6_no_pseudocount / self.Neff)  


    #Compute and return the n-site marginals with fixed site positions for generic n
    def compute_fn_no_pseudocount(self, n, indices): 
        shape_list = self.q * np.ones(2*n, dtype = int)
        shape_list[::2]  = 1
        fn_no_pseudocount = np.zeros(shape_list)
        for i_seq, sequence in enumerate(self.label_to_seq.values()):
            which_component = [self.aag_dict[sequence[indices[i//2]]] if i%2 else 0 for i in range(2*n)]
            fn_no_pseudocount[tuple(which_component)] += self.weights[i_seq]
        return (fn_no_pseudocount / self.Neff)  


    #Compute and return the 3-site connected correlations with two fixed site positions and one free
    def compute_CM3_1free(self, indices): 
        if self.CM2 is None:
            self.CM2 = self.compute_CM2()
        f3_no_pseudocount = self.compute_f3_no_pseudocount(indices)
        CM3 = np.zeros((1, self.q, 1, self.q, self.L, self.q))
        i_1 = indices[0]
        i_2 = indices[1]             
        for i_3 in range(self.L):
            for A_1 in range(self.q):
                for A_2 in range(self.q):
                    for A_3 in range(self.q):
                        CM3[0, A_1, 0, A_2, i_3, A_3] = (f3_no_pseudocount[0, A_1, 0, A_2, i_3, A_3] 
                                                        - self.f1_no_pseudocount[i_1, A_1] * self.CM2[i_2, A_2, i_3, A_3]
                                                        - self.f1_no_pseudocount[i_2, A_2] * self.CM2[i_1, A_1, i_3, A_3]
                                                        - self.f1_no_pseudocount[i_3, A_3] * self.CM2[i_1, A_1, i_2, A_2]
                                                        - self.f1_no_pseudocount[i_1, A_1] * self.f1_no_pseudocount[i_2, A_2] * self.f1_no_pseudocount[i_3, A_3]
                                                        )
        return CM3   
        

    #Compute and return the 4-site connected correlations with fixed site positions
    def compute_CM4(self, indices):
        if self.CM2 is None:
            self.CM2 = self.compute_CM2()
        f4_no_pseudocount = self.compute_f4_no_pseudocount(indices)
        CM4 = np.zeros((1, self.q, 1, self.q, 1, self.q, 1, self.q))

        i_1 = indices[0]
        i_2 = indices[1]
        i_3 = indices[2]
        i_4 = indices[3]

        g123 = self.compute_CM3([i_1, i_2, i_3])
        g124 = self.compute_CM3([i_1, i_2, i_4])
        g134 = self.compute_CM3([i_1, i_3, i_4])
        g234 = self.compute_CM3([i_2, i_3, i_4])
                
        for A_1 in range(self.q):
            for A_2 in range(self.q):
                for A_3 in range(self.q):
                    for A_4 in range(self.q):
                        part_1 = (self.f1_no_pseudocount[i_1, A_1] * g234[0, A_2, 0, A_3, 0, A_4] +
                                  self.f1_no_pseudocount[i_2, A_2] * g134[0, A_1, 0, A_3, 0, A_4] +
                                  self.f1_no_pseudocount[i_3, A_3] * g124[0, A_1, 0, A_2, 0, A_4] +
                                  self.f1_no_pseudocount[i_4, A_4] * g123[0, A_1, 0, A_2, 0, A_3]
                        )
                        part_2 = (self.CM2[i_1, A_1, i_2, A_2] * self.CM2[i_3, A_3, i_4, A_4] +
                                  self.CM2[i_1, A_1, i_3, A_3] * self.CM2[i_2, A_2, i_4, A_4] +
                                  self.CM2[i_1, A_1, i_4, A_4] * self.CM2[i_2, A_2, i_3, A_3]        
                        )
                        part_3 =(self.f1_no_pseudocount[i_1, A_1]*self.f1_no_pseudocount[i_2, A_2]*self.CM2[i_3, A_3, i_4, A_4] +
                                  self.f1_no_pseudocount[i_1, A_1]*self.f1_no_pseudocount[i_3, A_3]*self.CM2[i_2, A_2, i_4, A_4] +
                                  self.f1_no_pseudocount[i_1, A_1]*self.f1_no_pseudocount[i_4, A_4]*self.CM2[i_2, A_2, i_3, A_3] +
                                  self.f1_no_pseudocount[i_2, A_2]*self.f1_no_pseudocount[i_3, A_3]*self.CM2[i_1, A_1, i_4, A_4] +
                                  self.f1_no_pseudocount[i_2, A_2]*self.f1_no_pseudocount[i_4, A_4]*self.CM2[i_1, A_1, i_3, A_3] +
                                  self.f1_no_pseudocount[i_3, A_3]*self.f1_no_pseudocount[i_4, A_4]*self.CM2[i_1, A_1, i_2, A_2] 
                        )
                        part_4 = (self.f1_no_pseudocount[i_1, A_1]*self.f1_no_pseudocount[i_2, A_2]
                                  *self.f1_no_pseudocount[i_3, A_3]*self.f1_no_pseudocount[i_4, A_4])

                        CM4[0, A_1, 0, A_2, 0, A_3, 0, A_4] = (f4_no_pseudocount[0, A_1, 0, A_2, 0, A_3, 0, A_4]  
                                                                    - part_1 - part_2 - part_3 - part_4)
        return CM4  
    

    #Auxiliary function to save the marginals and correlations in the instance to a folder          
    def save_correlations(self, file_name):
        np.save(file_name + "_f1_no_pseudocount.npy", self.f1_no_pseudocount)
        np.save(file_name + "_f2_no_pseudocount.npy", self.f2_no_pseudocount)
        np.save(file_name + "_f1.npy", self.f1)
        np.save(file_name + "_f2.npy", self.f2)
        np.save(file_name + "_CM2.npy", self.CM2) 
        #np.save(file_name + "_covariance_matrix.npy", self.CM)     
        #np.save(file_name + "_couplings_e.npy", self.E)
        #np.save(file_name + "_direct_information.npy", self.DI)
        np.save(file_name + "_mutual_information.npy", self.MI)



























