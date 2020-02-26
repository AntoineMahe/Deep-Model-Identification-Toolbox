# MODEL SEARCH
network_list="/home/gt_ari/systemid_v2/scripts/network_list/ATTNMPMH_networks_LRELU.txt"

# VALIDATION PARAMETERS
run_max=2

# TRAINING PARAMETERS
iterations=10000
used_points=12
pred_points=1
batch_size=64
batch_test_size=10000
batch_val_size=10000
drop_rate=0.1
learning_rate=0.001
traj_length=20

# PATH
root="/home/gt_ari/systemid_v2"
root2="/scratch/gt_ari"
training_set="$root2/data_RSS/simu/train-full-random"
validation_set="$root2/data_RSS/simu/val"
test_set="$root2/data_RSS/simu/test"
python_script="$root/networks/run_training.py"

# OUTPUT
output_root="/scratch/gt_ari/RSS-2020-Models"
to_file="$output_root/model_search_attnmpmh_full_random.txt"
save_suffix="$output_root/results/GS/ATTNMPMH_full_random"
tb_suffix="$output_root/tensorboard/GS/ATTNMPMH_full_random"

# INIT
mkdir -p ${output_root}

while IFS= read -r model        
do
	for run in $(seq 0 $run_max)
	do
		mkdir -p ${save_suffix}/${model}/r${run}
		echo python3 ${python_script} --train_data ${training_set} --val_data ${validation_set} --test_data ${test_set} --batch_size ${batch_size} --val_batch_size ${batch_val_size} --test_batch_size ${batch_test_size} --input_dim 5 --output_dim 3 --dropout ${drop_rate} --model ${model} --learning_rate ${learning_rate} --timestamp_idx 0 --output ${save_suffix}/${model}/r${run} --tb_dir ${tb_suffix} --tb_log_name ${model}-r${run} --reader_mode seq2seq --trajectory_length=${traj_length} --max_iterations ${iterations} >> ${to_file}
	done
done < "$network_list"
