export MSA_data_folder='./data/MSA'
export MSA_weights_location='./data/weights'
export VAE_checkpoint_location='./results/VAE_parameters'
export training_logs_location='./logs/'
export protein_index=0
export all_singles_mutations_folder='./data/mutations'
export output_evol_indices_location='./results/evol_indices'
export num_samples_compute_evol_indices=20000
export batch_size=2048
export computation_mode='all_singles'

export MSA_list='./data/mappings/mapping_PABP.csv'
export model_name_suffix='38k'
export model_parameters_location='./EVE/default_model_params.json'
export checkpoint_file_name='PABP_38k/PABP_38k_step_400000'
export output_evol_indices_filename_suffix='400'



python compute_evol_indices.py \
    --MSA_data_folder ${MSA_data_folder} \
    --MSA_list ${MSA_list} \
    --protein_index ${protein_index} \
    --MSA_weights_location ${MSA_weights_location} \
    --VAE_checkpoint_location ${VAE_checkpoint_location} \
    --model_name_suffix ${model_name_suffix} \
    --model_parameters_location ${model_parameters_location} \
    --checkpoint_file_name ${checkpoint_file_name} \
    --computation_mode ${computation_mode} \
    --all_singles_mutations_folder ${all_singles_mutations_folder} \
    --output_evol_indices_location ${output_evol_indices_location} \
    --num_samples_compute_evol_indices ${num_samples_compute_evol_indices} \
    --batch_size ${batch_size} \
    --output_evol_indices_filename_suffix  ${output_evol_indices_filename_suffix}