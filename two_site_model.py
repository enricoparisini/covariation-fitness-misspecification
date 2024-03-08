from correlations import *
from utils.msa_manipulation import *
from utils.various_tools import *


#Auxiliary function to compute 1-site fields given a 2-site couplings and 1-site marginals.
@jit(nopython=True)
def _compute_1_site_fields(J_ij, f_i):
    L, q = f_i.shape
    hi = np.zeros((L, q))
    for i in range(L):
        log_fi = np.log(f_i[i] / f_i[i, q - 1])
        J_ij_sum = np.zeros((1, q))
        for j in range(L):
            if i != j:
                J = J_ij[i, :, j]
                J_ij_sum += np.dot(
                    J, f_i[j].reshape((1, q)).T
                ).T
        hi[i] = log_fi - J_ij_sum
    return hi


class Two_site_model(Correlations):
    def __init__(self, correlations_instance_location=None, father=None, label_to_seq=None, weights=None, 
                 E_couplings_location = None, E_couplings = None, is_indep = False, theta=0.2, pseudocount=0.5):
        '''
        This class represents a 2-site model: it contains the MSA and the marginals it is trained on, the 1-site fields and the 2-site couplings, 
        as well as methods to sample sequences from the model and to generate evolutionary indices for given sequences.
        It can be initialised based on a Correlations instance (from either its location on disk or the instance itself), 
        or from the MSA in dictionary label_to_seq format. In either cases, one can define the 2-site couplings at initialisation and if so,
        1-site fields are then fitted at initialisation using 1-site marginals.
        Parameters:
        - correlations_instance_location: (string) path to the Correlations instance used to initialise the Two_site_model instance. 
            If provided, the parameters 'father', 'label_to_seq' and 'weights' are ignored.
        - father: Correlations instance used to initialise the Two_site_model instance. If provided, 'label_to_seq' and 'weights' parameters
            are ignored.
        - label_to_seq: (dict) MSA the 2-site model is trained on.
        - weights: (np array) weights of each sequence in the alignment based on similarity.
        - E_couplings_location: (string) location of the 2-site couplings of the model. To be provided either with shape (L, q, L, q), 
            or with the flattened shape used by Mi3. In the latter case, the couplings are automatically converted to shape (L, q, L, q).
        - E_couplings: (np.array) 2-site couplings of the model. To be provided either with shape (L, q, L, q), or with the flattened shape used by Mi3.
            In the latter case, the couplings are automatically converted to shape (L, q, L, q).
        - is_indep: (bool) if True, the couplings are set to zero. If provided, the parameters 'E_couplings_location' and 'E_couplings' are ignored.
        - theta: (float) sequence weighting hyperparameter.
        - pseudocount: (float) parameter estimating the uncertainty in marginals due to finite-sample effects.
        '''
        self.h1 = None
        if correlations_instance_location is not None:
            father = load_instance_from_file(correlations_instance_location)
        if father is not None:
            super().__init__(label_to_seq=father.label_to_seq, weights=father.weights, theta=father.theta, pseudocount=father.pseudocount)
            self.load_from_Correlations_instance(father=father)
        else:
            super().__init__(label_to_seq=label_to_seq, weights=weights, theta=theta, pseudocount=pseudocount)   
        if E_couplings_location is not None:
            E_couplings = np.load(E_couplings_location)
        if E_couplings is not None:
            if E_couplings.shape[0] > self.L:
                self.E = self.converted_Mi3_couplings(E_couplings)
                self.h1 = np.zeros((self.L, self.q))
            else:
                self.E = E_couplings
                self.compute_1_site_fields()
        if is_indep:
            self.E = np.zeros((self.L, self.q, self.L, self.q))
            self.compute_1_site_fields()


    #Convert the 2-site couplings from the flattened form provided by Mi3 as output to shape (L, q, L, q)
    def converted_Mi3_couplings(self, J):
        E = np.zeros((self.L, self.q, self.L, self.q))
        for index_ij, (i,j) in enumerate([(i,j) for i in range(self.L-1) for j in range(i+1,self.L)]):
            for index_ab, (a,b) in enumerate([(a , b) for a in range(self.q) for b in range(self.q)]):
                E[i, a, j, b] = -J[index_ij, index_ab]
                E[j, b, i, a] = -J[index_ij, index_ab]
        return E


    #Fit 1-site fields appearing in the Hamiltonian of the 2-site model based on 2-site couplings and 1-point marginals
    def compute_1_site_fields(self):
        self.h1 = _compute_1_site_fields(self.E, self.f1)


    #Function to load marginals and MSA from a Correlations instance at a later stage than initialisation
    def load_from_Correlations_instance(self, correlations_instance_location=None, father=None):
        if correlations_instance_location is not None:
            father = load_instance_from_file(correlations_instance_location)
        self.weights = father.weights
        self.label_to_seq = father.label_to_seq
        if father.label_to_OHE is not None:
            self.label_to_OHE = father.label_to_OHE
        if father.f1_no_pseudocount is not None:
            self.f1_no_pseudocount = father.f1_no_pseudocount 
        if father.f2_no_pseudocount is not None:
            self.f2_no_pseudocount = father.f2_no_pseudocount 
        if father.f1 is not None:
            self.f1 = father.f1
        if father.f2 is not None:
            self.f2 = father.f2
        if father.CM is not None:
            self.CM = father.CM
        if father.CM2 is not None:
            self.CM2 = father.CM2
        if father.MI is not None:
            self.MI = father.MI
        if father.E is not None:
            self.E = father.E
        if father.DI is not None:
            self.DI = father.DI


    #Compute the energy of a sequence, which can be provided either as a list of characters
    #or as a list of indices (referring to positions within the AA dictionary 'self.aag_dict')
    def energy_of_sequence(self, seq):
        if type(seq[0]) == str:
            seq = seq_to_indices(seq, self.aag_dict)
        if len(seq) != self.L:
            raise Exception("This sequence is not of the right length.")     
        exponent_1_site, exponent_2_site = 0, 0
        for i_letter_1 in range(self.L):
            exponent_1_site += self.h1[i_letter_1, seq[i_letter_1]]
            for i_letter_2 in range(i_letter_1, self.L):
                exponent_2_site += self.E[i_letter_1, seq[i_letter_1],i_letter_2, seq[i_letter_2]] 
        return (exponent_1_site + exponent_2_site)
    

    def compute_evol_indices_single_mutations(self, list_mutations, offset):
        '''
        Compute the evolutionary indices for a list of mutations.
        Parameters:
        - list_mutations: (pd.DataFrame) contains the mutations of the MSA focus sequence in the format provided by EVE.
        - offset: (int) the position within the focus protein of the first site in the MSA.
        '''
        wt_energy = self.energy_of_sequence(next(iter(self.label_to_seq.values())))
        focus_seq = next(iter(self.label_to_seq.values())).copy()
        relative_energies = np.zeros(len(list_mutations)) 
        for i_mutation, mutation in enumerate(list_mutations['mutations']):
            mutated_sequence = focus_seq.copy()
            _, pos, mut_aa = mutation[0], int(mutation[1:-1])-offset, mutation[-1]
            mutated_sequence[pos] = mut_aa
            relative_energies[i_mutation] = self.energy_of_sequence(mutated_sequence)      
        return wt_energy - relative_energies


    def metropolis_hastings_sampler(self, n_iter = 10000):
        '''
        Sample a sequence from the 2-site model using a Metropolis-Hastings algorithm for 'n_iter' iterations.
        The returned data is a list consisting of
        - seq: (list) the sequence as a list of indices (referring to positions within the AA dictionary 'self.aag_dict').
        - seq_0: (list) the starting sequence of the algorithm as a list of indices (referring to positions within the AA dictionary 'self.aag_dict').
        - energies: (list) contains the 'n_iter' energies of the sequences produced by the algorithm.
        - list_seqs: (list) contains the 'n_iter' sequences produced by the algorithm, each as a list of indices 
            (referring to positions within the AA dictionary 'self.aag_dict').
        '''
        seq_0 = np.random.choice(self.q, size = self.L)
        seq = seq_0.copy()
        prob_0 = np.exp(self.energy_of_sequence(seq),dtype=np.float128)   
        energies = [np.log(prob_0)]
        list_seqs = [seq_0]
        
        for _ in range(n_iter): 
            site_position = np.random.choice(self.L)
            original_letter = seq[site_position]        
            proposed_letter = np.random.choice(self.q, size = 1)
            if proposed_letter == original_letter:
                continue      
            seq[site_position] = proposed_letter
            prob_prop = np.exp(self.energy_of_sequence(seq),dtype=np.longdouble)         
            accept_prob = min(1, prob_prop / prob_0)
            u = np.random.uniform()
            if(u < accept_prob):
                prob_0 = prob_prop
                energies += [np.log(prob_prop)]
            else:
                seq[site_position] = original_letter
                energies += [np.log(prob_0)]
            list_seqs += [np.copy(seq)]      
        return [seq, seq_0, energies, list_seqs]



















