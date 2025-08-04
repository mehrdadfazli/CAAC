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
from src.CAAC_utils import SelfAttentionModifier, compute_attention_factor
from transformers.cache_utils import Cache

def recorder(out):
    NEG_WORDS = ["No", "not", "no", "NO"]
    out = out.replace('.', '').replace(',', '')
    words = out.split(' ')
    if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
        return "No"
    else:
        return "Yes"


# Set up logging
parser = argparse.ArgumentParser(description="Run AMBER benchmark with hallucination mitigation")
parser.add_argument("--model_type", default="instructblip", choices=["instructblip", "llava", "qwen2-vl", "llava-next"], help="Model type")
parser.add_argument("--load_in_8bit", action="store_true", default=True, help="Load model in 8-bit precision")
parser.add_argument("--cache_dir", default="/scratch/mfazli/huggingface_cache", help="Cache directory for model")
parser.add_argument("--amber_path", default="/projects/zzhu20/Mehrdad/AMBER", help="Path to AMBER dataset")
parser.add_argument("--log_dir", default="/projects/zzhu20/Mehrdad/lvlm_hallucination/Results/AMBER/April-20-25", help="Directory for logs and results")
parser.add_argument("--use_CAAC", action="store_true", default=False, help="Use CAAC for hallucination mitigation")
parser.add_argument("--do_sample", action="store_true", default=False, help="Use sampling for generation")
parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for generation")
parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
parser.add_argument("--img_txt_cal_layers", type=int, nargs="+", default=list(range(32)), help="Layers for upscaling intervention")
parser.add_argument("--img_cal_layers", type=int, nargs="+", default=[0, 1], help="Layers for calibration intervention")
parser.add_argument("--min_lamb", type=float, default=1.0, help="Minimum lambda for attention scaling")
parser.add_argument("--max_lamb", type=float, default=1.5, help="Maximum lambda for attention scaling")
parser.add_argument("--confidence_threshold", type=float, default=0.25, help="Confidence threshold for dynamic scaling")
parser.add_argument("--calibration_query", default="nana pina sequ zuta rupi", help="Query for calibration")
parser.add_argument("--input_token_idx_calibration", type=int, nargs="+", default=[-1, -2, -3], help="Token indices for calibration")
parser.add_argument("--ref_image", default="self", choices=["self", "white", "black", "noise"], help="Reference image for calibration")
parser.add_argument("--beta", type=float, default=0, help="Beta parameter for calibration")
args = parser.parse_args()

os.makedirs(args.log_dir, exist_ok=True)
LOG_FILE = os.path.join(args.log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Derived paths
EXP_ID = np.random.randint(1000,9999)
JSON_QUERY_PATH = os.path.join(args.amber_path, "data/query/query_generative.json")
JSON_ANNOTATION_PATH = os.path.join(args.amber_path, "data/annotations.json")
IMAGE_DIR = os.path.join(args.amber_path, "image")
EXP_CONFIG_PATH = os.path.join(args.log_dir, f"{EXP_ID}_{args.model_type}_config.json")
RESPONSES_PATH = os.path.join(args.log_dir, f"{EXP_ID}_{args.model_type}.json")

# Model names
model_names = {
    "instructblip": "Salesforce/instructblip-vicuna-7b",
    "llava": "llava-hf/llava-1.5-7b-hf",
    "qwen2-vl": "Qwen/Qwen2-VL-7B-Instruct",
    "llava-next": "llava-hf/llava-v1.6-vicuna-7b-hf"
}


# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Experiment configuration
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
    "max_new_tokens": args.max_new_tokens
}

# Save EXP_CONFIG to JSON
config_to_save = {k: v for k, v in EXP_CONFIG.items() if k != "compute_attention_factor"}
try:
    with open(EXP_CONFIG_PATH, 'w') as file:
        json.dump(config_to_save, file, indent=4)
    logger.info(f"Saved experiment configuration to {EXP_CONFIG_PATH}")
except Exception as e:
    logger.error(f"Failed to save experiment configuration: {e}")
    raise

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
                    # next_token = model.generate(
                    #     **inputs,
                    #     do_sample=args.do_sample,
                    #     max_new_tokens=1,
                    #     # num_beams=args.num_beams
                    # )[:, -1:]
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
        return generated[:, input_length:]
    
    except Exception as e:
        logger.error(f"Error in dynamic_generate: {e}")
        raise
    
def dynamic_generate_slow(raw_image, query, exp_config, model, processor, model_type):
    inputs = process_inputs(raw_image, query, processor, model_type)
    generated = inputs["input_ids"]
    input_length = generated.shape[-1]

    if model_type == "qwen2-vl":
        img_token_idxs = torch.nonzero(inputs['input_ids'][0] == model.config.image_token_id, as_tuple=False).flatten().cpu()
    else:
        img_token_idxs = torch.nonzero(inputs['input_ids'][0] == model.config.image_token_index, as_tuple=False).flatten().cpu()

    modifier = SelfAttentionModifier(model, exp_config, img_token_idxs)
    attention_maps = {}

    # Construct white image with same size as input
    white_image = Image.new("RGB", raw_image.size, (255, 255, 255))

    with torch.no_grad():
        modifier.update_calibration_matrix(attention_maps, white_image, processor, model_type)

    modifier.register_hooks()
    
    with torch.no_grad():
        for _ in range(exp_config['max_new_tokens']):
            inputs["input_ids"] = generated
            inputs["attention_mask"] = torch.ones_like(generated)
            outputs = model(**inputs)
            last_logits = outputs.logits[:, -1, :]
            confidence = torch.max(torch.nn.functional.softmax(last_logits, dim=-1)).item()

            if confidence > exp_config['confidence_threshold']:
                next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
            else:
                modifier.dynamic_factor = exp_config['compute_attention_factor'](exp_config, confidence)
                next_token = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=1,
                    num_beams=1
                )[:, -1:]

            generated = torch.cat([generated, next_token], dim=-1)
            modifier.dynamic_factor = exp_config['compute_attention_factor'](exp_config, 1)

            if next_token.item() == processor.tokenizer.eos_token_id:
                break

            torch.cuda.empty_cache()

    modifier.remove_hooks()
    return generated[:, input_length:]


def main():
    model, processor = load_model_and_processor(args.model_type, model_names, args.cache_dir, device, args.load_in_8bit)
    tokenizer = processor.tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    with open(JSON_QUERY_PATH, 'r') as file:
        data = json.load(file)
    with open(JSON_ANNOTATION_PATH, 'r') as file:
        annotations = json.load(file)

    responses = []
    if os.path.exists(RESPONSES_PATH):
        with open(RESPONSES_PATH, 'r') as file:
            responses = json.load(file)

    processed_ids = set(item['id'] for item in responses)


    for item in tqdm(data, desc="Processing Dataset"):
        if item['id'] in processed_ids:
            continue

        img_path = os.path.join(IMAGE_DIR, item['image'])
        try:
            raw_image = Image.open(img_path).convert('RGB')
        except:
            continue

        query = item['query']

        if args.use_CAAC:
            generated_ids = dynamic_generate(raw_image, query, EXP_CONFIG, model, processor, args.model_type)
            response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            response_text_64 = processor.batch_decode(generated_ids[:, :64], skip_special_tokens=True)[0].strip()

            if item['id'] <= 1004:
                responses.append({
                    'id': item['id'],
                    'response_64': response_text_64,
                    # 'response_128': response_text[:128],
                    # 'response_256': response_text[:256],
                    # 'response_512': response_text[:512],
                    'response': response_text,
                    'response_length': generated_ids.shape[-1]
                })
            else:
                responses.append({
                    'id': item['id'],
                    'response': recorder(response_text),
                    # 'response_length': generated_ids.shape[-1]
                })

        else:
            inputs = process_inputs(raw_image, query, processor, args.model_type)
            generated_ids = model.generate(**inputs, do_sample=args.do_sample, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
            response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            # response_text_64 = processor.batch_decode(generated_ids[:, :64], skip_special_tokens=True)[0].strip()
            
            response_text = response_text.split("ASSISTANT: ")[1]
            # response_text_64 = response_text_64.split("ASSISTANT: ")[1]

            if item['id'] <= 1004:
                responses.append({
                    'id': item['id'],
                    # 'response_64': response_text_64,
                    # 'response_128': response_text[:128],
                    # 'response_256': response_text[:256],
                    # 'response_512': response_text[:512],
                    'response': response_text,
                    'response_length': generated_ids.shape[-1]
                })
            else:
                responses.append({
                    'id': item['id'],
                    'response': recorder(response_text),
                    # 'response_length': generated_ids.shape[-1]
                })

        processed_ids.add(item['id'])
        with open(RESPONSES_PATH, 'w') as file:
            json.dump(responses, file, indent=4)
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()