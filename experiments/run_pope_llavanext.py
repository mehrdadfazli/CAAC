import os
import gc
import json
import torch
import argparse
import logging
from datetime import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm
from src.model_utils import load_model_and_processor, process_inputs
from src.CAAC_utils import SelfAttentionModifier, compute_attention_factor

# Set up argument parsing
parser = argparse.ArgumentParser(description="Evaluate LLaVA-NeXT with CAAC on POPE benchmark")
parser.add_argument("--model_type", type=str, default="llava-next", choices=["llava-next"], help="Model type (fixed to llava-next)")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--question_dir", default="/projects/zzhu20/Mehrdad/CAG/datasets/POPE/coco", help="Directory containing POPE question JSON files")
parser.add_argument("--image_folder", default="/projects/zzhu20/Mehrdad/CAG/datasets/POPE/coco/val2014", help="Path to image folder")
parser.add_argument("--log_dir", default="./results/POPE_llava-next", help="Directory for logs and results")
parser.add_argument("--load_in_8bit", action="store_true", default=True, help="Load model in 8-bit precision")
parser.add_argument("--cache_dir", default="/scratch/mfazli/huggingface_cache", help="Cache directory for model")
parser.add_argument("--use_CAAC", action="store_true", default=False, help="Use CAAC for hallucination mitigation")
parser.add_argument("--do_sample", action="store_true", default=False, help="Use sampling for generation")
parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for generation")
parser.add_argument("--max_new_tokens", type=int, default=10, help="Maximum new tokens to generate")
parser.add_argument("--img_txt_cal_layers", type=int, nargs="+", default=list(range(32)), help="Layers for upscaling intervention")
parser.add_argument("--img_cal_layers", type=int, nargs="+", default=[0, 1], help="Layers for calibration intervention")
parser.add_argument("--min_lamb", type=float, default=1.0, help="Minimum lambda for attention scaling")
parser.add_argument("--max_lamb", type=float, default=1.5, help="Maximum lambda for attention scaling")
parser.add_argument("--confidence_threshold", type=float, default=0.25, help="Confidence threshold for dynamic scaling")
parser.add_argument("--calibration_query", default="nana pina sequ zuta rupi", help="Query for calibration")
parser.add_argument("--input_token_idx_calibration", type=int, nargs="+", default=[-1, -2, -3], help="Token indices for calibration")
parser.add_argument("--ref_image", default="white", choices=["self", "white", "black", "noise"], help="Reference image for calibration")
parser.add_argument("--beta", type=float, default=0, help="Beta parameter for calibration")
args = parser.parse_args()

# Define all prompting strategies
STRATEGIES = ["adversarial", "popular", "random"]

# Set up logging
model_log_dir = os.path.join(args.log_dir, args.model_type)
os.makedirs(model_log_dir, exist_ok=True)
LOG_FILE = os.path.join(model_log_dir, f"log_all_strategies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Model names
model_names = {
    "llava-next": "llava-hf/llava-v1.6-vicuna-7b-hf"
}

# Device setup
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Experiment configuration
EXP_ID = np.random.randint(1000, 9999)
EXP_CONFIG_PATH = os.path.join(model_log_dir, f"{EXP_ID}_{args.model_type}_config.json")
EXP_CONFIG = {
    "img_txt_cal_layers": args.img_txt_cal_layers,
    "img_cal_layers": args.img_cal_layers,
    "compute_attention_factor": compute_attention_factor,
    "min_lamb": args.min_lamb,
    "max_lamb": args.max_lamb,
    "confidence_threshold": args.confidence_threshold,
    "calibration_query": args.calibration_query,
    "input_token_idx_calibration": args.input_token_idx_calibration,
    "ref_image": args.ref_image,
    "beta": args.beta,
    "max_new_tokens": args.max_new_tokens,
    "strategies": STRATEGIES
}

# Save experiment configuration
config_to_save = {k: v for k, v in EXP_CONFIG.items() if k != "compute_attention_factor"}
try:
    with open(EXP_CONFIG_PATH, 'w') as file:
        json.dump(config_to_save, file, indent=4)
    logger.info(f"Saved experiment configuration to {EXP_CONFIG_PATH}")
except Exception as e:
    logger.error(f"Failed to save experiment configuration: {e}")
    raise

# Global calibration cache
calibration_cache = {}

def dynamic_generate(raw_image, query, exp_config, model, processor, model_type):
    """
    Custom generation loop with dynamic attention scaling for LLaVA-NeXT, optimized with past_key_values.
    """
    try:
        # Process initial inputs
        inputs = process_inputs(raw_image, query, processor, model_type)
        generated = inputs["input_ids"]
        input_length = generated.shape[-1]
        current_input_ids = generated
        attention_mask = inputs["attention_mask"]

        # Fix: Only compute pixel_values and image_sizes once
        pixel_values = inputs["pixel_values"]
        image_sizes = inputs["image_sizes"]

        # Identify image token indices
        img_token_idxs = torch.nonzero(
            inputs['input_ids'][0] == (
                model.config.image_token_id if model_type == "qwen2-vl" 
                else model.config.image_token_index
            ), 
            as_tuple=False
        ).flatten().cpu()

        # Initialize modifier and calibration
        modifier = SelfAttentionModifier(model, exp_config, img_token_idxs)
        attention_maps = {}
        if exp_config['ref_image'] == 'self':
            ref_image = raw_image
        elif exp_config['ref_image'] == 'white':
            ref_image = Image.new("RGB", raw_image.size, (255, 255, 255))
        elif exp_config['ref_image'] == 'black':
            ref_image = Image.new("RGB", raw_image.size, (0, 0, 0))
        elif exp_config['ref_image'] == 'noise':
            ref_image = Image.fromarray(np.random.randint(0, 256, raw_image.size + (3,), dtype=np.uint8), mode="RGB")

        with torch.no_grad():
            calibration_inputs = process_inputs(ref_image, exp_config['calibration_query'], processor, model_type)
            modifier.update_calibration_matrix(attention_maps, ref_image, processor, model_type)
        attention_maps.clear()

        modifier.register_hooks()
        n_repeats_forward = 0
        past_key_values = None

        with torch.no_grad():
            for _ in range(exp_config['max_new_tokens']):
                # Prepare inputs for the current step
                inputs_for_forward = {
                    "input_ids": current_input_ids,
                    "attention_mask": attention_mask,
                    "pixel_values": pixel_values,
                    "image_sizes": image_sizes,
                    "past_key_values": past_key_values,
                    "use_cache": True
                }

                # Forward pass to compute confidence
                outputs = model(**inputs_for_forward)
                last_logits = outputs.logits[:, -1, :]
                confidence = torch.max(torch.nn.functional.softmax(last_logits, dim=-1)).item()

                if confidence > exp_config['confidence_threshold']:
                    next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
                    past_key_values = outputs.past_key_values
                else:
                    n_repeats_forward += 1
                    modifier.dynamic_factor = exp_config['compute_attention_factor'](exp_config, confidence)
                    outputs = model(**inputs_for_forward)  # second forward with modified attention
                    last_logits = outputs.logits[:, -1, :]
                    next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
                    past_key_values = outputs.past_key_values

                # Append the new token
                generated = torch.cat([generated, next_token], dim=-1)

                # Update inputs for the next step
                current_input_ids = next_token
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)],
                    dim=-1
                )

                # Reset dynamic factor for next step
                modifier.dynamic_factor = exp_config['compute_attention_factor'](exp_config, 1)

                # Check for EOS token
                if next_token.item() == processor.tokenizer.eos_token_id:
                    break

                torch.cuda.empty_cache()

        modifier.remove_hooks()
        return generated[:, input_length:], n_repeats_forward / max(1, generated[:, input_length:].shape[-1])
    
    except Exception as e:
        logger.error(f"Error in dynamic_generate: {e}")
        raise

def process_strategy(strategy, model, processor, args):
    """
    Process a single POPE strategy (adversarial, popular, random).
    """
    try:
        # Derive question file and answers path
        question_file = os.path.join(args.question_dir, f"coco_pope_{strategy}.json")
        answers_path = os.path.join(model_log_dir, f"{EXP_ID}_{strategy}.jsonl")
        
        # Load questions
        with open(question_file, 'r') as file:
            questions = [json.loads(line) for line in file]
        logger.info(f"Loaded {len(questions)} questions from {question_file}")
        
        # Process questions and save answers
        with open(answers_path, 'w') as ans_file:
            for item in tqdm(questions, desc=f"Processing {strategy} Questions"):
                question_id = item["question_id"]
                image_file = item["image"]
                qs = item["text"]
                prompt = qs + " Please answer this question with one word."
                img_path = os.path.join(args.image_folder, image_file)
                if not os.path.exists(img_path):
                    logger.warning(f"Image {img_path} not found, skipping")
                    continue
                try:
                    raw_image = Image.open(img_path).convert('RGB')
                except Exception as e:
                    logger.error(f"Error loading image {img_path}: {e}")
                    continue
                
                if args.use_CAAC:
                    generated_ids, n_repeats_forward = dynamic_generate(raw_image, prompt, EXP_CONFIG, model, processor, args.model_type)
                    response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                else:
                    inputs = process_inputs(raw_image, prompt, processor, args.model_type)
                    generated_ids = model.generate(
                        **inputs,
                        do_sample=args.do_sample,
                        max_new_tokens=args.max_new_tokens,
                        num_beams=args.num_beams
                    )
                    response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                    response_text = response_text.split("ASSISTANT: ")[1] if "ASSISTANT: " in response_text else response_text
                    n_repeats_forward = 0
                
                ans_file.write(json.dumps({
                    "question_id": question_id,
                    "prompt": prompt,
                    "text": response_text,
                    "model_id": args.model_type,
                    "image": image_file,
                    "n_repeats_forward": n_repeats_forward,
                    "metadata": {}
                }) + "\n")
                ans_file.flush()
                
                torch.cuda.empty_cache()
                gc.collect()
                if len(calibration_cache) > 5:
                    calibration_cache.clear()
        
        logger.info(f"Saved answers for {strategy} to {answers_path}")
    except Exception as e:
        logger.error(f"Error processing strategy {strategy}: {e}")
        raise

def main():
    """
    Main function to evaluate LLaVA-NeXT with CAAC on the POPE benchmark.
    """
    try:
        # Load model and processor
        logger.info(f"Loading model: {args.model_type}")
        model, processor = load_model_and_processor(args.model_type, model_names, args.cache_dir, device, args.load_in_8bit)
        model.eval()
        logger.info("Model and processor loaded successfully")

        if args.use_CAAC:
            logger.info(f"using CAAC")

        # Process each strategy
        for strategy in STRATEGIES:
            process_strategy(strategy, model, processor, args)
        
        logger.info(f"Evaluation complete. Results saved in {model_log_dir}")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()