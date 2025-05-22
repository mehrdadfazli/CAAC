#!/bin/bash

# Get the configuration file and exp_dir from arguments
CONFIG_FILE=$1

# Function to read a single value from JSON
read_config() {
    python -c "import json; config=json.load(open('$CONFIG_FILE')); print(config.get('$1', ''))"
}

# Function to read an array from JSON
read_array() {
    python -c "import json; config=json.load(open('$CONFIG_FILE')); print(' '.join(map(str, config.get('$1', []))))"
}

# Read variables
model_type=$(read_config model_type)
load_in_8bit=$(read_config load_in_8bit)
cache_dir=$(read_config cache_dir)
amber_path=$(read_config amber_path)
do_sample=$(read_config do_sample)
use_CAG=$(read_config use_CAG)
num_beams=$(read_config num_beams)
log_dir=$(read_config log_dir)
max_new_tokens=$(read_config max_new_tokens)
img_txt_cal_layers=($(read_array img_txt_cal_layers))
img_cal_layers=($(read_array img_cal_layers))
min_lamb=$(read_config min_lamb)
max_lamb=$(read_config max_lamb)
confidence_threshold=$(read_config confidence_threshold)
calibration_query=$(read_config calibration_query)
input_token_idx_calibration=($(read_array input_token_idx_calibration))
ref_image=$(read_config ref_image)
beta=$(read_config beta)

# Check if model_type is read correctly
if [ -z "$model_type" ]; then
    echo "Error: Failed to read 'model_type' from $CONFIG_FILE"
    exit 1
fi


# Build the command
cmd="python ../run_amber.py \
    --model_type $model_type \
    --cache_dir $cache_dir \
    --amber_path $amber_path \
    --log_dir $log_dir \
    --num_beams $num_beams \
    --max_new_tokens $max_new_tokens \
    --img_txt_cal_layers ${img_txt_cal_layers[*]} \
    --img_cal_layers ${img_cal_layers[*]} \
    --min_lamb $min_lamb \
    --max_lamb $max_lamb \
    --confidence_threshold $confidence_threshold \
    --input_token_idx_calibration ${input_token_idx_calibration[*]} \
    --ref_image $ref_image \
    --beta $beta"

# Add boolean flags conditionally
if [ "$load_in_8bit" = "true" ]; then
    cmd="$cmd --load_in_8bit"
fi
if [ "$do_sample" = "true" ]; then
    cmd="$cmd --do_sample"
fi
if [ "$use_CAG" = "true" ]; then
    cmd="$cmd --use_CAG"
fi

# Execute the command
eval "$cmd"