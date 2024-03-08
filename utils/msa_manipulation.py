import tqdm
import numpy as np

from correlations import *
from utils.various_tools import *


def generate_MSA_from_MSA(input_MSA_filename, output_MSA_filename, stride = 1, shift = 0, N_out = 1e8, list_keep_sequences = [0]):
    '''
    Generates a new, smaller MSA from a larger MSA.
    Parameters:
        stride: controls the stride when going through the input MSA.
        shift: the first sequence where the input parsing starts.
        N_out: maximum number of output sequences after considering shift and stride.
        list_keep_sequences: list of sequences that are kept regardless of the other parameters; 
                             by default, only the input focus sequence is kept.
    '''
    def cond_i_label(i_label):
        #An auxiliary function implementing the condition for a sequence to be written to output
        return (list_keep_sequences is not None and i_label in list_keep_sequences) or ((not (i_label)%stride) and ( i_label > shift))

    with open(input_MSA_filename) as input_file:
        lines = np.array(input_file.read().splitlines())
    i_label = -1
    n_written = 0
    with open(output_MSA_filename, "w") as output_file: 
        for line in lines:
            if line[0] == ">":
                i_label += 1
                if cond_i_label(i_label):
                    n_written += 1
                    if n_written > N_out:
                        break
                    else:
                        output_file.write(line + '\n')      
            else:
                if cond_i_label(i_label):
                    output_file.write(line + '\n')


def save_labeltoseq_to_MSA(label_to_seq, output_MSA_filename):
    #Saves a dictionary label_to_seq to an MSA
    with open(output_MSA_filename, 'w') as file:
        for label, sequence in label_to_seq.items():
            file.write(label + '\n' + "".join(sequence) + '\n')


def convert_MSA_EVE_to_Mi3(label_to_seq, output_MSA_filename, num_spaces = 10):
    '''
    Converts MSA from the dictionary label_to_seq format used by EVE to the format required by Mi3.
    num_spaces: number of spaces between each label and sequence in the output file.
    '''
    #identify longest label
    lengths = np.array(list(map(len, label_to_seq.keys())))
    longest_label = list(label_to_seq.keys())[np.argmax(lengths)]
    len_longest = len(longest_label)
    #write each sequence to file
    with open(output_MSA_filename, 'w') as file:
        for label, sequence in label_to_seq.items():
            file.write(label + ' '*(len_longest - len(label) + num_spaces) + "".join(sequence) + '\n')


def convert_MSA_Mi3_generated_to_EVE(input_MSA_filename, output_MSA_filename, focus_line, N_out = 10000, prefix = ">Seq_"):
    '''
    Converts MSA generated by Mi3 from the format used by Mi3 to the default format used by EVE.
    Parameters:
        focus_line: the focus line in string form to be prepended to the output MSA.
        N_out: defines how many sequences are kept from the Mi3 MSA. 
                  It implicitly determines the stride used to select the output sequences
        prefix: prefix in string form of generated sequences labels.
    '''
    with open(input_MSA_filename) as input_file:
        lines = np.array(input_file.read().splitlines())
    stride = len(lines)//N_out
    lines = lines[::stride][:N_out-1]
    with open(output_MSA_filename, "w") as output_file:
        output_file.write(focus_line + '\n')
        for i_label, line in enumerate(lines):
            output_file.write(prefix + str(i_label) + '\n' + line + '\n')


def find_furthest_sequences(label_to_seq_whole, label_to_seq_subset, output_MSA_filename, N_out = 10000,
                            keep_focus_seq_whole = True):
    '''
    Given a large MSA and a subset MSA, it writes to file the N_out 
    (Hamming-)furthest sequences of the large MSA with respect to the subset MSA.
    Parameters:
        label_to_seq_whole: large MSA in dictionary label_to_seq format.
        label_to_seq_subset: subset MSA in dictionary label_to_seq format.
        output_MSA_filename:  string for the output MSA file name.
        N_out: number of output sequences.
        keep_focus_seq_whole: bool forcing the focus sequence of the large MSA to be prepended to the output.
    Returns:
        the output MSA in dictionary label_to_seq format,
        the indices of the sequences kept from the input MSA.
    '''
    if keep_focus_seq_whole:
        N_out -= 1 
    model_whole = Correlations(label_to_seq = label_to_seq_whole)
    model_training = Correlations(label_to_seq = label_to_seq_subset)
    model_whole._OHE_MSA()
    model_training._OHE_MSA()
    label_to_OHE_whole_reshaped = model_whole.label_to_OHE.reshape(model_whole.Neff,-1)
    label_to_OHE_training_reshaped = model_training.label_to_OHE.reshape(model_training.Neff,-1)
    list_candidates = np.zeros((model_whole.Neff))

    for sequence in tqdm.tqdm(label_to_OHE_training_reshaped):
        prod = np.dot(label_to_OHE_whole_reshaped, sequence)
        indices_furthest = np.argpartition(prod, N_out)[:N_out]
        list_candidates[indices_furthest] += 1
    indices_final_candidates = np.argpartition(list_candidates, -N_out)[-N_out:]

    if keep_focus_seq_whole:
        label_to_seq_final_candidates = {next(iter(model_whole.label_to_seq.keys())) : next(iter(model_whole.label_to_seq.values()))}
    else:
        label_to_seq_final_candidates = {}
    label_to_seq_final_candidates.update({key: value for index, (key, value) in enumerate(model_whole.label_to_seq.items())
                                     if index in indices_final_candidates})
    save_labeltoseq_to_MSA(label_to_seq_final_candidates, output_MSA_filename)
    return [label_to_seq_final_candidates, indices_final_candidates]


def initial_and_final_label_to_seq_focus(label_to_seq, N_letters = 10):
    #Shows the initial and final N_letters of the focus sequence 
    #of an MSA in dictionary label_to_seq format
    print(f"Initial {N_letters} letters: \t{next(iter(label_to_seq.values()))[:N_letters]}")
    print(f"Final   {N_letters} letters:  \t{next(iter(label_to_seq.values()))[-N_letters:]}")


def indices_to_seq(indices, alphabet):
    #Converts a list of indices into a list of letters according to alphabet
    return np.array(list(alphabet))[indices]


def seq_to_indices(seq, letterg_dict):
    #Converts a list of letters into a list of indices according to alphabet
    return np.array([letterg_dict[letter] for letter in seq])