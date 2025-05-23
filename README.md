# README

## Project Description
This repository contains an implementation focused on evaluating and benchmarking vision-language models using the AMBER, CHAIR, and POPE datasets. The project aims to assess the performance of these models on various multimodal tasks, with an emphasis on understanding and improving their capabilities, such as reducing hallucination and enhancing factual accuracy in generated outputs. It provides a robust framework for researchers and developers to conduct experiments and analyze results in a structured and reproducible manner.

## Implementation Details
Our implementation leverages the Hugging Face versions of LLaVA-1.5 and InstructBLIP, built on Transformer version 4.47. These models form the backbone of the project, enabling efficient evaluation and benchmarking across the specified datasets.

## Dataset Requirements
To run the experiments, you need to download the following datasets:

- **AMBER Dataset**: Please download the AMBER dataset from its original repository [AMBER](https://github.com/junyangwang0410/AMBER). Ensure it is extracted and accessible on your system.
- **MS COCO 2014 Validation Set**: Required for the POPE and CHAIR benchmarks. Download the validation set from the official [MS COCO](https://cocodataset.org/#home) website and prepare it for use in the experiments.

## Environment Setup
To set up the required environment, follow these steps:

1. **Install Conda**: Ensure you have Miniconda or Anaconda installed on your system. If not, download and install it from the official website.
2. **Create Conda Environment**: Create a new Conda environment named `CAAC` with Python 3.10 by running:
   ```bash
   conda create -n CAAC python=3.10
   ```
3. **Activate Environment**: Activate the environment with:
   ```bash
   conda activate CAAC
   ```
4. **Install Dependencies**: Install the required dependencies listed in the `requirements.txt` file by running:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration
Before running the experiments, you must configure the `config.json` file located in the `configs` folder. Update the following parameters with the appropriate paths:

- **`cache_dir`**: The directory where model checkpoints will be stored.
- **`amber_path`**: The path to the downloaded AMBER dataset.
- **`chair_path`**: The path to the CHAIR dataset.
- **`POPE_question_dir`**: The path to the POPE dataset questions.
- **`POPE_image_folder`**: The path to the POPE dataset images.
- **`log_dir`**: The directory where experiment results will be saved.

Ensure all paths are absolute or correctly relative to the execution directory to avoid runtime errors.

## Running Experiments
To execute the experiments, use the provided shell scripts with the configured `config.json` file. Run the following commands from the root of the repository:

- **For CHAIR Experiments**:
  ```bash
  bash run_chair.sh ../../configs/config.json
  ```
- **For AMBER Experiments**:
  ```bash
  bash run_amber.sh ../../configs/config.json
  ```
- **For POPE Experiments**:
  ```bash
  bash run_pope.sh ../../configs/config.json
  ```

Make sure the `config.json` file is properly set up before running these commands. The scripts will process the respective benchmarks and save the results to the directory specified in `log_dir`.

## Logs Directory
The `logs` directory contains experimental results for the AMBER, CHAIR, and POPE benchmarks, copied from the CAAC framework. These logs provide detailed insights into the model's performance across each benchmark. Each log file corresponds to a specific experiment and includes metrics such as hallucination rates, accuracy, and other relevant scores.

## Evals Directory
The `evals` directory contains scripts used to evaluate model outputs on the AMBER, CHAIR, and POPE benchmarks. Below are the details and usage instructions for each script:

### 1. `pope.py`
This script evaluates the model's outputs on the POPE benchmark.

**Usage**:
```bash
python pope.py --gt_files /path/to/POPE/coco_pope_popular.json --gen_files /path/to/POPE_output_file.json
```
- **`--gt_files`**: Path to the ground truth JSON file for POPE.
- **`--gen_files`**: Path to the generated outputs JSON file from your model.

### 2. `chair.py`
This script evaluates the model's outputs on the CHAIR benchmark.

**Usage**:
```bash
python chair.py --coco_path /path/to/CHAIR/annotations --cap_file /path/to/CHAIR_output_file.jsonl
```
- **`--coco_path`**: Path to the COCO annotations directory.
- **`--cap_file`**: Path to the generated captions file in JSONL format.

### 3. `inference.py`
This script evaluates the model's outputs on the AMBER benchmark. It is a slightly modified version of the original AMBER evaluation script.

**Usage**:
```bash
python inference.py --inference_data /path/to/your/inference/file --evaluation_type g --gen_response_tag response_512
```
- **`--inference_data`**: Path to the inference data file.
- **`--evaluation_type`**: Type of evaluation (e.g., `g` for generative tasks).
- **`--gen_response_tag`**: Tag for the generated response (e.g., `response_512` for responses with up to 512 tokens).

**Note**: Update the paths in the commands to match your local file system. For additional options or details, refer to the script documentation or source code.

## Additional Notes
- Verify that all dataset paths are correctly specified in `config.json` to prevent issues during execution.
- The repository structure includes folders such as `configs/` for configuration files,󠁧 `scripts/` for shell scripts, `logs/` for experimental results, and `evals/` for evaluation scripts. If your structure differs, adjust the paths in the commands accordingly.
- For further assistance or to report issues, please refer to the repository’s documentation or contact the maintainers.

Happy benchmarking!