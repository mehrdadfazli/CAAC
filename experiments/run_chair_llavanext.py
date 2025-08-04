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

# Set up logging
parser = argparse.ArgumentParser(description="Evaluate LLaVA-NeXT with CAAC on CHAIR benchmark")
parser.add_argument("--model_type", type=str, default="llava-next", choices=["llava-next"], help="Model type (fixed to llava-next)")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--data_path", type=str, default="../CAG/datasets/coco2014/val2014/", help="Path to COCO 2014 validation images")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size (fixed to 1)")
parser.add_argument("--load_in_8bit", action="store_true", default=True, help="Load model in 8-bit precision")
parser.add_argument("--cache_dir", default="/scratch/mfazli/huggingface_cache", help="Cache directory for model")
parser.add_argument("--log_dir", default="./results/CHAIR_llava-next", help="Directory for logs and results")
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
parser.add_argument("--ref_image", default="white", choices=["self", "white", "black", "noise"], help="Reference image for calibration")
parser.add_argument("--beta", type=float, default=0, help="Beta parameter for calibration")
parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--opera_results", type=str, default=None, help="Path to OPERA results JSONL file to read image IDs from")
parser.add_argument("--num_images", type=int, default=500, help="Number of images to process when not using opera_results")
args = parser.parse_args()

# Create log directory
model_log_dir = os.path.join(args.log_dir, args.model_type)
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
EXP_CONFIG_PATH = os.path.join(model_log_dir, f"{EXP_ID}_{args.model_type}_config.json")
RESPONSES_PATH = os.path.join(model_log_dir, f"{EXP_ID}_{args.model_type}.jsonl")

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
        return generated[:, input_length:], n_repeats_forward / max(1, generated[:, input_length:].shape[-1])
    
    except Exception as e:
        logger.error(f"Error in dynamic_generate: {e}")
        raise

def main():
    """
    Main function to evaluate LLaVA-NeXT with CAAC on the CHAIR benchmark.
    """
    try:
        # Load model and processor
        logger.info(f"Loading model: {args.model_type}")
        model, processor = load_model_and_processor(args.model_type, model_names, args.cache_dir, device, args.load_in_8bit)
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
            raw_image = Image.open(img_path).convert("RGB")
            query = "Please describe this image in detail."
            
            if args.use_CAAC:
                # generated_ids, n_repeats_forward = dynamic_generate_slow(raw_image, query, EXP_CONFIG, model, processor, args.model_type)
                generated_ids, n_repeats_forward = dynamic_generate(raw_image, query, EXP_CONFIG, model, processor, args.model_type)
                caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            else:
                inputs = process_inputs(raw_image, query, processor, args.model_type)
                generated_ids = model.generate(
                    **inputs,
                    do_sample=args.do_sample,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams
                )
                caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                caption = caption.split("ASSISTANT: ")[1] if "ASSISTANT: " in caption else caption
                n_repeats_forward = 0

            img_save = {
                "image_id": img_id,
                "caption": caption,
                "n_repeats_forward": n_repeats_forward
            }
            with open(RESPONSES_PATH, "a") as f:
                json.dump(img_save, f)
                f.write('\n')
            torch.cuda.empty_cache()
            gc.collect()
            if len(calibration_cache) > 5:
                calibration_cache.clear()

        logger.info(f"Evaluation complete. Results saved to {RESPONSES_PATH}")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()


#======================================================================================================================================

# def dynamic_generate_slow(raw_image, query, exp_config, model, processor, model_type):
#     """
#     Custom generation loop with dynamic attention scaling for LLaVA-NeXT.
    
#     Args:
#         raw_image: PIL Image object
#         query: Input text query
#         exp_config: Dictionary with CAAC configuration
#         model: Hugging Face model instance
#         processor: Hugging Face processor instance
#         model_type: String indicating model type (e.g., 'llava-next')
    
#     Returns:
#         tuple: (generated token IDs excluding input tokens, n_repeats_forward)
#     """
#     try:
#         inputs = process_inputs(raw_image, query, processor, model_type)
#         generated = inputs["input_ids"]
#         input_length = generated.shape[-1]

#         img_token_idxs = torch.nonzero(
#             inputs['input_ids'][0] == (
#                 model.config.image_token_id if model_type == "qwen2-vl" 
#                 else model.config.image_token_index
#             ), 
#             as_tuple=False
#         ).flatten().cpu()
#         logger.info(f"Image token count: {len(img_token_idxs)}")

#         modifier = SelfAttentionModifier(model, exp_config, img_token_idxs)
#         attention_maps = {}

#         # Construct reference image based on ref_image argument
#         if exp_config['ref_image'] == 'self':
#             ref_image = raw_image
#         elif exp_config['ref_image'] == 'white':
#             ref_image = Image.new("RGB", raw_image.size, (255, 255, 255))
#         elif exp_config['ref_image'] == 'black':
#             ref_image = Image.new("RGB", raw_image.size, (0, 0, 0))
#         elif exp_config['ref_image'] == 'noise':
#             ref_image = Image.fromarray(np.random.randint(0, 256, raw_image.size + (3,), dtype=np.uint8), mode="RGB")

#         # Cache calibration matrix
#         image_size = raw_image.size
#         with torch.no_grad():
#             modifier.update_calibration_matrix(attention_maps, ref_image, processor, model_type)

#         # if image_size not in calibration_cache:
#         #     with torch.no_grad():
#         #         modifier.update_calibration_matrix(attention_maps, ref_image, processor, model_type)
#         #     calibration_cache[image_size] = attention_maps
#         # else:
#         #     attention_maps = calibration_cache[image_size]

#         attention_maps.clear()

#         modifier.register_hooks()
#         n_repeats_forward = 0

#         with torch.no_grad():
#             for _ in range(exp_config['max_new_tokens']):
#                 inputs["input_ids"] = generated
#                 inputs["attention_mask"] = torch.ones_like(generated)
#                 outputs = model(**inputs)
#                 last_logits = outputs.logits[:, -1, :]
#                 confidence = torch.max(torch.nn.functional.softmax(last_logits, dim=-1)).item()

#                 if confidence > exp_config['confidence_threshold']:
#                     next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
                    
#                 else:
#                     n_repeats_forward += 1
#                     modifier.dynamic_factor = exp_config['compute_attention_factor'](exp_config, confidence)
#                     next_token = model.generate(
#                         **inputs,
#                         do_sample=args.do_sample,
#                         max_new_tokens=1,
#                         num_beams=args.num_beams
#                     )[:, -1:]

#                 generated = torch.cat([generated, next_token], dim=-1)
#                 modifier.dynamic_factor = exp_config['compute_attention_factor'](exp_config, 1)

#                 if next_token.item() == processor.tokenizer.eos_token_id:
#                     break

#                 torch.cuda.empty_cache()

#         modifier.remove_hooks()
#         return generated[:, input_length:], n_repeats_forward / generated[:, input_length:].shape[-1]
    
#     except Exception as e:
#         logger.error(f"Error in dynamic_generate: {e}")
#         raise

#======================================================================================================================================

# import os
# import gc
# import json
# import torch
# import random
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# import argparse
# import logging
# from datetime import datetime
# from src.model_utils import load_model_and_processor, process_inputs
# from src.CAAC_utils import SelfAttentionModifier, compute_attention_factor

# # Set up logging
# parser = argparse.ArgumentParser(description="Evaluate LLaVA-NeXT with CAAC on CHAIR benchmark")
# parser.add_argument("--model_type", type=str, default="llava-next", choices=["llava-next"], help="Model type (fixed to llava-next)")
# parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
# parser.add_argument("--data_path", type=str, default="../CAG/datasets/coco2014/val2014/", help="Path to COCO 2014 validation images")
# parser.add_argument("--batch_size", type=int, default=1, help="Batch size (fixed to 1)")
# parser.add_argument("--load_in_8bit", action="store_true", default=True, help="Load model in 8-bit precision")
# parser.add_argument("--cache_dir", default="/scratch/mfazli/huggingface_cache", help="Cache directory for model")
# parser.add_argument("--log_dir", default="./results/CHAIR_llava-next", help="Directory for logs and results")
# parser.add_argument("--use_CAAC", action="store_true", default=False, help="Use CAAC for hallucination mitigation")
# parser.add_argument("--do_sample", action="store_true", default=False, help="Use sampling for generation")
# parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for generation")
# parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
# parser.add_argument("--img_txt_cal_layers", type=int, nargs="+", default=list(range(32)), help="Layers for upscaling intervention")
# parser.add_argument("--img_cal_layers", type=int, nargs="+", default=[0, 1], help="Layers for calibration intervention")
# parser.add_argument("--min_lamb", type=float, default=1.0, help="Minimum lambda for attention scaling")
# parser.add_argument("--max_lamb", type=float, default=1.5, help="Maximum lambda for attention scaling")
# parser.add_argument("--confidence_threshold", type=float, default=0.25, help="Confidence threshold for dynamic scaling")
# parser.add_argument("--calibration_query", default="nana pina sequ zuta rupi", help="Query for calibration")
# parser.add_argument("--input_token_idx_calibration", type=int, nargs="+", default=[-1, -2, -3], help="Token indices for calibration")
# parser.add_argument("--ref_image", default="white", choices=["self", "white", "black", "noise"], help="Reference image for calibration")
# parser.add_argument("--beta", type=float, default=0, help="Beta parameter for calibration")
# parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
# parser.add_argument("--opera_results", type=str, default=None, help="Path to OPERA results JSONL file to read image IDs from")
# parser.add_argument("--num_images", type=int, default=500, help="Number of images to process when not using opera_results")
# args = parser.parse_args()

# # Create log directory
# model_log_dir = os.path.join(args.log_dir, args.model_type)
# os.makedirs(model_log_dir, exist_ok=True)
# LOG_FILE = os.path.join(model_log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
# file_handler = logging.FileHandler(LOG_FILE)
# file_handler.setLevel(logging.INFO)
# file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
# logger.addHandler(file_handler)

# # Derived paths
# EXP_ID = np.random.randint(1000, 9999)
# EXP_CONFIG_PATH = os.path.join(model_log_dir, f"{EXP_ID}_{args.model_type}_config.json")
# RESPONSES_PATH = os.path.join(model_log_dir, f"{EXP_ID}_{args.model_type}.jsonl")

# # Model names
# model_names = {
#     "llava-next": "llava-hf/llava-v1.6-vicuna-7b-hf"
# }

# # Device setup
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# logger.info(f"Using device: {device}")

# # Experiment configuration
# EXP_CONFIG = {
#     "img_txt_cal_layers": args.img_txt_cal_layers,
#     "img_cal_layers": args.img_cal_layers,
#     "compute_attention_factor": compute_attention_factor,
#     "min_lamb": args.min_lamb,
#     "max_lamb": args.max_lamb,
#     "confidence_threshold": args.confidence_threshold,
#     "calibration_query": args.calibration_query,
#     "input_token_idx_calibration": args.input_token_idx_calibration,
#     "ref_image": args.ref_image,
#     "beta": args.beta,
#     "max_new_tokens": args.max_new_tokens
# }

# # Save experiment configuration
# config_to_save = {k: v for k, v in EXP_CONFIG.items() if k != "compute_attention_factor"}
# try:
#     with open(EXP_CONFIG_PATH, 'w') as file:
#         json.dump(config_to_save, file, indent=4)
#     logger.info(f"Saved experiment configuration to {EXP_CONFIG_PATH}")
# except Exception as e:
#     logger.error(f"Failed to save experiment configuration: {e}")
#     raise

# # Global calibration cache
# calibration_cache = {}

# def load_opera_image_ids(opera_results_path):
#     image_ids = []
#     try:
#         with open(opera_results_path, 'r') as f:
#             for line in f:
#                 data = json.loads(line)
#                 image_ids.append(data['image_id'])
#         return image_ids
#     except Exception as e:
#         logger.error(f"Error reading OPERA results file {opera_results_path}: {e}")
#         raise

# # def dynamic_generate(raw_image, query, exp_config, model, processor, model_type):
# #     """
# #     Custom generation loop with dynamic attention scaling for LLaVA-NeXT.
    
# #     Args:
# #         raw_image: PIL Image object
# #         query: Input text query
# #         exp_config: Dictionary with CAAC configuration
# #         model: Hugging Face model instance
# #         processor: Hugging Face processor instance
# #         model_type: String indicating model type (e.g., 'llava-next')
    
# #     Returns:
# #         tuple: (generated token IDs excluding input tokens, n_repeats_forward)
# #     """
# #     try:
# #         inputs = process_inputs(raw_image, query, processor, model_type)
# #         generated = inputs["input_ids"]
# #         input_length = generated.shape[-1]

# #         img_token_idxs = torch.nonzero(
# #             inputs['input_ids'][0] == (
# #                 model.config.image_token_id if model_type == "qwen2-vl" 
# #                 else model.config.image_token_index
# #             ), 
# #             as_tuple=False
# #         ).flatten().cpu()
# #         logger.info(f"Image token count: {len(img_token_idxs)}")

# #         modifier = SelfAttentionModifier(model, exp_config, img_token_idxs)
# #         attention_maps = {}

# #         # Construct reference image based on ref_image argument
# #         if exp_config['ref_image'] == 'self':
# #             ref_image = raw_image
# #         elif exp_config['ref_image'] == 'white':
# #             ref_image = Image.new("RGB", raw_image.size, (255, 255, 255))
# #         elif exp_config['ref_image'] == 'black':
# #             ref_image = Image.new("RGB", raw_image.size, (0, 0, 0))
# #         elif exp_config['ref_image'] == 'noise':
# #             ref_image = Image.fromarray(np.random.randint(0, 256, raw_image.size + (3,), dtype=np.uint8), mode="RGB")

# #         # Cache calibration matrix
# #         image_size = raw_image.size
# #         with torch.no_grad():
# #             modifier.update_calibration_matrix(attention_maps, ref_image, processor, model_type)

# #         # if image_size not in calibration_cache:
# #         #     with torch.no_grad():
# #         #         modifier.update_calibration_matrix(attention_maps, ref_image, processor, model_type)
# #         #     calibration_cache[image_size] = attention_maps
# #         # else:
# #         #     attention_maps = calibration_cache[image_size]

# #         attention_maps.clear()

# #         modifier.register_hooks()
# #         n_repeats_forward = 0

# #         with torch.no_grad():
# #             for _ in range(exp_config['max_new_tokens']):
# #                 inputs["input_ids"] = generated
# #                 inputs["attention_mask"] = torch.ones_like(generated)
# #                 outputs = model(**inputs)
# #                 last_logits = outputs.logits[:, -1, :]
# #                 confidence = torch.max(torch.nn.functional.softmax(last_logits, dim=-1)).item()

# #                 if confidence > exp_config['confidence_threshold']:
# #                     next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
                    
# #                 else:
# #                     n_repeats_forward += 1
# #                     modifier.dynamic_factor = exp_config['compute_attention_factor'](exp_config, confidence)
# #                     next_token = model.generate(
# #                         **inputs,
# #                         do_sample=args.do_sample,
# #                         max_new_tokens=1,
# #                         num_beams=args.num_beams
# #                     )[:, -1:]

# #                 generated = torch.cat([generated, next_token], dim=-1)
# #                 modifier.dynamic_factor = exp_config['compute_attention_factor'](exp_config, 1)

# #                 if next_token.item() == processor.tokenizer.eos_token_id:
# #                     break

# #                 torch.cuda.empty_cache()

# #         modifier.remove_hooks()
# #         return generated[:, input_length:], n_repeats_forward / generated[:, input_length:].shape[-1]
    
# #     except Exception as e:
# #         logger.error(f"Error in dynamic_generate: {e}")
# #         raise


# def dynamic_generate(raw_image, query, exp_config, model, processor, model_type):
#     """
#     Optimized CAAC generation for HF Transformers v4.47+ using new Cache format and proper image patch handling.
#     """
#     try:
#         # ===== Fix #1: Ensure processor has correct vision config =====
#         if model_type == "llava-next":
#             processor.patch_size = model.config.vision_config.patch_size
#             processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy

#         # ===== Step 1: Encode prompt and image =====
#         inputs = process_inputs(raw_image, query, processor, model_type)
#         input_ids = inputs["input_ids"]  # shape: [1, T]
#         attention_mask = torch.ones_like(input_ids)
#         input_length = input_ids.shape[-1]

#         # ===== Step 2: Locate image tokens =====
#         img_token_idxs = torch.nonzero(
#             input_ids[0] == (
#                 model.config.image_token_id if model_type == "qwen2-vl"
#                 else model.config.image_token_index
#             ), as_tuple=False
#         ).flatten().cpu()
#         logger.info(f"Image token count: {len(img_token_idxs)}")

#         # ===== Step 3: Set up modifier for CAAC =====
#         modifier = SelfAttentionModifier(model, exp_config, img_token_idxs)
#         attention_maps = {}

#         ref_image = {
#             "self": raw_image,
#             "white": Image.new("RGB", raw_image.size, (255, 255, 255)),
#             "black": Image.new("RGB", raw_image.size, (0, 0, 0)),
#             "noise": Image.fromarray(np.random.randint(0, 256, raw_image.size + (3,), dtype=np.uint8), mode="RGB")
#         }[exp_config["ref_image"]]

#         with torch.no_grad():
#             modifier.update_calibration_matrix(attention_maps, ref_image, processor, model_type)
#         attention_maps.clear()
#         modifier.register_hooks()

#         # ===== Step 4: Initialize generation =====
#         generated_tokens = []
#         next_input_ids = input_ids
#         past_key_values = None
#         n_repeats_forward = 0

#         # Initialize cache using HF's method (returns a Cache object)
#         model_inputs = model.prepare_inputs_for_generation(
#             input_ids=next_input_ids,
#             attention_mask=attention_mask,
#             use_cache=True
#         )

#         outputs = model(**model_inputs)
#         logits = outputs.logits[:, -1, :]
#         cache = outputs.past_key_values  # HF Cache object

#         for step in range(exp_config['max_new_tokens']):
#             confidence = torch.max(torch.nn.functional.softmax(logits, dim=-1)).item()

#             # Update dynamic attention factor if needed
#             if confidence <= exp_config["confidence_threshold"]:
#                 modifier.dynamic_factor = exp_config["compute_attention_factor"](exp_config, confidence)
#                 n_repeats_forward += 1
#             else:
#                 modifier.dynamic_factor = exp_config["compute_attention_factor"](exp_config, 1.0)

#             # Select next token
#             next_token = torch.argmax(logits, dim=-1, keepdim=True)  # shape: [1, 1]
#             if next_token.item() == processor.tokenizer.eos_token_id:
#                 break
#             generated_tokens.append(next_token)

#             # Update input and attention
#             attention_mask = torch.cat([
#                 attention_mask,
#                 torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)
#             ], dim=-1)

#             # Feed the new token with cache
#             outputs = model(
#                 input_ids=next_token,
#                 attention_mask=attention_mask,
#                 past_key_values=cache,
#                 use_cache=True
#             )
#             logits = outputs.logits[:, -1, :]
#             cache = outputs.past_key_values

#         modifier.remove_hooks()

#         # Finalize sequence
#         if generated_tokens:
#             output_ids = torch.cat(generated_tokens, dim=-1)
#         else:
#             output_ids = torch.empty((1, 0), dtype=torch.long, device=input_ids.device)

#         repeat_ratio = n_repeats_forward / max(1, output_ids.shape[-1])
#         return output_ids, repeat_ratio

#     except Exception as e:
#         logger.error(f"Error in dynamic_generate: {e}")
#         raise




# def main():
#     """
#     Main function to evaluate LLaVA-NeXT with CAAC on the CHAIR benchmark.
#     """
#     try:
#         # Load model and processor
#         logger.info(f"Loading model: {args.model_type}")
#         model, processor = load_model_and_processor(args.model_type, model_names, args.cache_dir, device, args.load_in_8bit)
#         model.eval()
#         logger.info("Model and processor loaded successfully")
#         logger.info(f"OPERA results path: {args.opera_results}")

#         # List all images
#         if args.opera_results:
#             image_ids = load_opera_image_ids(args.opera_results)
#             image_list = [(f"COCO_val2014_{id:012d}.jpg", id) for id in image_ids]
#             logger.info(f"Loaded {len(image_list)} image IDs from OPERA results")
#         else:
#             img_files = [f for f in os.listdir(args.data_path) if f.endswith('.jpg')]
#             image_list = [(f, int(f.split(".jpg")[0][-6:])) for f in img_files]
#             random.seed(args.random_seed)
#             random.shuffle(image_list)
#             if args.num_images is not None:
#                 image_list = image_list[:args.num_images]
#             logger.info(f"Processing {len(image_list)} images from {args.data_path}")

#         # Process each image
#         for img_file, img_id in tqdm(image_list, desc="Generating captions"):
#             img_path = os.path.join(args.data_path, img_file)
#             if not os.path.exists(img_path):
#                 logger.warning(f"Image {img_path} not found, skipping")
#                 continue
#             # try:
#             raw_image = Image.open(img_path).convert("RGB")
#             query = "Please describe this image in detail."
            
#             if args.use_CAAC:
#                 generated_ids, n_repeats_forward = dynamic_generate(raw_image, query, EXP_CONFIG, model, processor, args.model_type)
#                 caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
#             else:
#                 processor.patch_size = model.config.vision_config.patch_size
#                 processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy
#                 inputs = process_inputs(raw_image, query, processor, args.model_type)
#                 generated_ids = model.generate(
#                     **inputs,
#                     do_sample=args.do_sample,
#                     max_new_tokens=args.max_new_tokens,
#                     num_beams=args.num_beams
#                 )
#                 caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
#                 caption = caption.split("ASSISTANT: ")[1] if "ASSISTANT: " in caption else caption
#                 n_repeats_forward = 0

#             img_save = {
#                 "image_id": img_id,
#                 "caption": caption,
#                 "n_repeats_forward": n_repeats_forward
#             }
#             with open(RESPONSES_PATH, "a") as f:
#                 json.dump(img_save, f)
#                 f.write('\n')
#             torch.cuda.empty_cache()
#             gc.collect()
#             # Clear calibration cache periodically
#             if len(calibration_cache) > 5:
#                 calibration_cache.clear()

#             # except Exception as e:
#             #     logger.error(f"Error processing image {img_file}: {e}")
#             #     continue

#         logger.info(f"Evaluation complete. Results saved to {RESPONSES_PATH}")

#     except Exception as e:
#         logger.error(f"Error in main: {e}")
#         raise

# if __name__ == "__main__":
#     main()