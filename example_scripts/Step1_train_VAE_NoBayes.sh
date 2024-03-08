export MSA_data_folder='./data/MSA'
export MSA_list='./data/mappings/mapping_PABP.csv'
export MSA_weights_location='./data/weights'
export VAE_checkpoint_location='./results/VAE_parameters/PABP_38k_NoBayes'
export model_name_suffix='38k_NoBayes'
export model_parameters_location='./EVE/default_model_params_NoBayes.json'
export training_logs_location='./logs/'
export protein_index=0
export load_model_location='./results/VAE_parameters/PABP_38k_NoBayes/PABP_38k_NoBayes_step_200000'

python train_VAE.py \
    --MSA_data_folder ${MSA_data_folder} \
    --MSA_list ${MSA_list} \
    --protein_index ${protein_index} \
    --MSA_weights_location ${MSA_weights_location} \
    --VAE_checkpoint_location ${VAE_checkpoint_location} \
    --model_name_suffix ${model_name_suffix} \
    --model_parameters_location ${model_parameters_location} \
    --training_logs_location ${training_logs_location} \
    --load_model_location ${load_model_location} \
