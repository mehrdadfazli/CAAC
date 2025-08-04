import os
import gc
import json
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import logging
from datetime import datetime
from src.model_utils import load_model_and_processor, process_inputs
from src.sampling_utils import generate_VCD, generate_M3ID, generate_AvisC

def recorder(out):
    """
    Convert response text to 'Yes' or 'No' based on presence of negative words.
    """
    NEG_WORDS = ["No", "not", "no", "NO"]
    out = out.replace('.', '').replace(',', '')
    words = out.split(' ')
    if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
        return "No"
    else:
        return "Yes"

# Set up argument parsing
parser = argparse.ArgumentParser(description="Evaluate LLaVA-NeXT with baseline methods on AMBER benchmark")
parser.add_argument("--model_type", type=str, default="llava-next", choices=["llava-next"], help="Model type (fixed to llava-next)")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--amber_path", default="/projects/zzhu20/Mehrdad/AMBER", help="Path to AMBER dataset")
parser.add_argument("--log_dir", default="./results/AMBER_llava-next/baselines", help="Directory for logs and results")
parser.add_argument("--load_in_8bit", action="store_true", default=True, help="Load model in 8-bit precision")
parser.add_argument("--cache_dir", default="/scratch/mfazli/huggingface_cache", help="Cache directory for model")
parser.add_argument("--use_VCD", action="store_true", default=False, help="Use VCD for hallucination mitigation")
parser.add_argument("--vcd_alpha", type=float, default=1.0, help="VCD alpha parameter")
parser.add_argument("--vcd_beta", type=float, default=0.1, help="VCD beta parameter")
parser.add_argument("--vcd_noise_step", type=int, default=500, help="VCD noise step parameter")
parser.add_argument("--use_M3ID", action="store_true", default=False, help="Use M3ID for hallucination mitigation")
parser.add_argument("--m3id_lamb", type=float, default=0.2, help="M3ID lambda parameter")
parser.add_argument("--m3id_beta", type=float, default=0.1, help="M3ID beta parameter")
parser.add_argument("--use_AvisC", action="store_true", default=False, help="Use AvisC for hallucination mitigation")
parser.add_argument("--avisc_alpha", type=float, default=2.5, help="AvisC alpha parameter")
parser.add_argument("--avisc_beta", type=float, default=0.1, help="AvisC beta parameter")
parser.add_argument("--avisc_layer_gamma", type=float, default=0.8, help="AvisC layer gamma parameter")
parser.add_argument("--avisc_lamb", type=float, default=1.0, help="AvisC lambda parameter")
parser.add_argument("--do_sample", action="store_true", default=False, help="Use sampling for generation")
parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for generation")
parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
args = parser.parse_args()

# Create log directory based on baseline method
if args.use_VCD:
    args.log_dir = os.path.join(args.log_dir, "VCD")
elif args.use_M3ID:
    args.log_dir = os.path.join(args.log_dir, "M3ID")
elif args.use_AvisC:
    args.log_dir = os.path.join(args.log_dir, "AvisC")
else:
    args.log_dir = os.path.join(args.log_dir, "base_model")

model_log_dir = args.log_dir
os.makedirs(model_log_dir, exist_ok=True)
LOG_FILE = os.path.join(model_log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Derived paths
EXP_ID = np.random.randint(1000, 9999)
JSON_QUERY_PATH = os.path.join(args.amber_path, "data/query/query_generative.json")
JSON_ANNOTATION_PATH = os.path.join(args.amber_path, "data/annotations.json")
IMAGE_DIR = os.path.join(args.amber_path, "image")
EXP_CONFIG_PATH = os.path.join(model_log_dir, f"{args.model_type}_{EXP_ID}_config.json")
RESPONSES_PATH = os.path.join(model_log_dir, f"{args.model_type}_{EXP_ID}.json")

# Model names
model_names = {
    "llava-next": "llava-hf/llava-v1.6-vicuna-7b-hf"
}

# Device setup
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Experiment configuration
EXP_CONFIG = {
    "use_VCD": str(args.use_VCD),
    "use_M3ID": str(args.use_M3ID),
    "use_AvisC": str(args.use_AvisC),
    "vcd_alpha": args.vcd_alpha,
    "vcd_beta": args.vcd_beta,
    "vcd_noise_step": args.vcd_noise_step,
    "m3id_lamb": args.m3id_lamb,
    "m3id_beta": args.m3id_beta,
    "avisc_alpha": args.avisc_alpha,
    "avisc_beta": args.avisc_beta,
    "avisc_layer_gamma": args.avisc_layer_gamma,
    "avisc_lamb": args.avisc_lamb,
    "max_new_tokens": args.max_new_tokens
}

# Save experiment configuration
with open(EXP_CONFIG_PATH, 'w') as file:
    json.dump(EXP_CONFIG, file, indent=4)
logger.info(f"Saved experiment configuration to {EXP_CONFIG_PATH}")

def main():
    """
    Main function to evaluate LLaVA-NeXT with baseline methods on the AMBER benchmark.
    """
    try:
        # Load model and processor
        logger.info(f"Loading model: {args.model_type}")
        model, processor = load_model_and_processor(args.model_type, model_names, args.cache_dir, device, args.load_in_8bit)
        tokenizer = processor.tokenizer
        model.eval()
        logger.info("Model and processor loaded successfully")

        # Load AMBER dataset
        with open(JSON_QUERY_PATH, 'r') as file:
            data = json.load(file)
        with open(JSON_ANNOTATION_PATH, 'r') as file:
            annotations = json.load(file)

        responses = []
        if os.path.exists(RESPONSES_PATH):
            with open(RESPONSES_PATH, 'r') as file:
                responses = json.load(file)

        processed_ids = set(item['id'] for item in responses)

        # Process each item in the dataset
        for item in tqdm(data, desc="Processing Dataset"):
            if item['id'] in processed_ids:
                continue

            img_path = os.path.join(IMAGE_DIR, item['image'])
            if not os.path.exists(img_path):
                logger.warning(f"Image {img_path} not found, skipping")
                continue
            try:
                raw_image = Image.open(img_path).convert('RGB')
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
                continue

            query = item['query']
            inputs = process_inputs(raw_image, query, processor, args.model_type)

            # Generate response based on selected baseline method
            if args.use_VCD:
                response = generate_VCD(
                    model=model,
                    tokenizer=tokenizer,
                    inputs=inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    raw_image=raw_image,
                    vcd_alpha=args.vcd_alpha,
                    vcd_beta=args.vcd_beta,
                    vcd_noise_step=args.vcd_noise_step
                )
            elif args.use_M3ID:
                response = generate_M3ID(
                    model=model,
                    tokenizer=tokenizer,
                    inputs=inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    raw_image=raw_image,
                    lamda=args.m3id_lamb,
                    beta=args.m3id_beta
                )
            elif args.use_AvisC:
                response = generate_AvisC(
                    model=model,
                    tokenizer=tokenizer,
                    inputs=inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    raw_image=raw_image,
                    avisc_alpha=args.avisc_alpha,
                    avisc_beta=args.avisc_beta,
                    layer_gamma=args.avisc_layer_gamma,
                    lamb=args.avisc_lamb
                )
            else:
                generated_ids = model.generate(
                    **inputs,
                    do_sample=args.do_sample,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams
                )
                response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                response = response.split("ASSISTANT: ")[1] if "ASSISTANT: " in response else response

            # Process response based on item ID
            if item['id'] <= 1004:
                responses.append({
                    'id': item['id'],
                    'response': response,
                    'response_length': len(tokenizer.encode(response, add_special_tokens=False))
                })
            else:
                responses.append({
                    'id': item['id'],
                    'response': recorder(response)
                })

            processed_ids.add(item['id'])
            with open(RESPONSES_PATH, 'w') as file:
                json.dump(responses, file, indent=4)
            torch.cuda.empty_cache()
            gc.collect()

        logger.info(f"Evaluation complete. Results saved to {RESPONSES_PATH}")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()