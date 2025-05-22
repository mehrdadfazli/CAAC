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

## Additional Notes
- Verify that all dataset paths are correctly specified in `config.json` to prevent issues during execution.
- The repository structure is assumed to include folders such as `configs/` for configuration files, `scripts/` for shell scripts, and `requirements.txt` for dependencies. If your structure differs, adjust the paths in the commands accordingly.
- For further assistance or to report issues, please refer to the repository’s documentation or contact the maintainers.

Happy benchmarking!