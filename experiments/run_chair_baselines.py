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

# Set up logging
parser = argparse.ArgumentParser(description="Evaluate LLaVA-NeXT with CAAC on CHAIR benchmark")
parser.add_argument("--model_type", type=str, default="llava-next", choices=["llava-next"], help="Model type (fixed to llava-next)")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--data_path", type=str, default="../CAG/datasets/coco2014/val2014/", help="Path to COCO 2014 validation images")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size (fixed to 1)")
parser.add_argument("--load_in_8bit", action="store_true", default=True, help="Load model in 8-bit precision")
parser.add_argument("--cache_dir", default="/scratch/mfazli/huggingface_cache", help="Cache directory for model")
parser.add_argument("--log_dir", default="./results/CHAIR_llava-next/baselines", help="Directory for logs and results")

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

parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--opera_results", type=str, default=None, help="Path to OPERA results JSONL file to read image IDs from")
parser.add_argument("--num_images", type=int, default=500, help="Number of images to process when not using opera_results")
args = parser.parse_args()

# Create log directory
if args.use_VCD:
    args.log_dir = os.path.join(args.log_dir, "VCD")
elif args.use_M3ID:
    args.log_dir = os.path.join(args.log_dir, "M3ID")
elif args.use_AvisC:
    args.log_dir = os.path.join(args.log_dir, "AvisC")
else:
    args.log_dir = os.path.join(args.log_dir, "base_model")

# model_log_dir = os.path.join(args.log_dir,args.model_type)
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
EXP_CONFIG_PATH = os.path.join(model_log_dir, f"{args.model_type}_{EXP_ID}_config.json")
RESPONSES_PATH  = os.path.join(model_log_dir, f"{args.model_type}_{EXP_ID}.jsonl")

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

def load_opera_image_ids(opera_results_path):
    image_ids = []
    try:
        with open(opera_results_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                image_ids.append(data['image_id'])
        return image_ids
    except Exception as e:
        logger.error(f"Error reading OPERA results file {opera_results_path}: {e}")
        raise


def main():
    """
    Main function to evaluate LLaVA-NeXT with CAAC on the CHAIR benchmark.
    """
    try:
        # Load model and processor
        logger.info(f"Loading model: {args.model_type}")
        model, processor = load_model_and_processor(args.model_type, model_names, args.cache_dir, device, args.load_in_8bit)
        tokenizer = processor.tokenizer
        model.eval()
        logger.info("Model and processor loaded successfully")
        logger.info(f"OPERA results path: {args.opera_results}")

        # List all images
        if args.opera_results:
            image_ids = load_opera_image_ids(args.opera_results)
            image_list = [(f"COCO_val2014_{id:012d}.jpg", id) for id in image_ids]
            logger.info(f"Loaded {len(image_list)} image IDs from OPERA results")
        else:
            img_files = [f for f in os.listdir(args.data_path) if f.endswith('.jpg')]
            image_list = [(f, int(f.split(".jpg")[0][-6:])) for f in img_files]
            random.seed(args.random_seed)
            random.shuffle(image_list)
            if args.num_images is not None:
                image_list = image_list[:args.num_images]
            logger.info(f"Processing {len(image_list)} images from {args.data_path}")

        # Process each image
        for img_file, img_id in tqdm(image_list, desc="Generating captions"):
            img_path = os.path.join(args.data_path, img_file)
            if not os.path.exists(img_path):
                logger.warning(f"Image {img_path} not found, skipping")
                continue
            # try:
            raw_image = Image.open(img_path).convert("RGB")
            query = "Please describe this image in detail."
            
            inputs = process_inputs(raw_image, query, processor, args.model_type)
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
                inputs = process_inputs(raw_image, query, processor, args.model_type)
                generated_ids = model.generate(
                    **inputs,
                    do_sample=args.do_sample,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams
                )
                response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                response = response.split("ASSISTANT: ")[1] if "ASSISTANT: " in response else response


            img_save = {
                "image_id": img_id,
                "caption": response
            }

            with open(RESPONSES_PATH, "a") as f:
                json.dump(img_save, f)
                f.write('\n')
            torch.cuda.empty_cache()
            gc.collect()

        logger.info(f"Evaluation complete. Results saved to {RESPONSES_PATH}")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()