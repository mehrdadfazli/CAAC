import argparse
import os
import json
import random
import torch
from PIL import Image
from tqdm import tqdm
import logging
from datetime import datetime
import numpy as np
import gc
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/src')

from model_utils import load_model_and_processor, process_inputs
from CAAC_utils import SelfAttentionModifier, compute_attention_factor


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define argument parser
parser = argparse.ArgumentParser(description="Evaluate on CHAIR with hallucination mitigation")
parser.add_argument("--model_type", type=str, default="instructblip", choices=["instructblip", "llava"], help="Model type")
parser.add_argument("--cache_dir", default="/path/to/model_checkpoints", help="Cache directory for model")
parser.add_argument("--chair_path", type=str, default="/path/to/coco2014/val2014/", help="Path to image dataset")
parser.add_argument("--log_dir", default="../results/CHAIR", help="Directory for logs and results")
parser.add_argument("--opera_results", type=str, default=None, help="Path to OPERA results JSONL file to read image IDs from. If none, then randomly selects 500 images.")
parser.add_argument("--num_images", type=int, default=500, help="Number of images to process when not using opera_results")
parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")

parser.add_argument("--img_txt_cal_layers", type=int, nargs="+", default=list(range(32)), help="Layers for upscaling intervention")
parser.add_argument("--img_cal_layers", type=int, nargs="+", default=list(range(10)), help="Layers for calibration intervention")
parser.add_argument("--min_lamb", type=float, default=1.0, help="Minimum lambda for attention scaling")
parser.add_argument("--max_lamb", type=float, default=1.5, help="Maximum lambda for attention scaling")
parser.add_argument("--confidence_threshold", type=float, default=0.25, help="Confidence threshold for dynamic scaling")
parser.add_argument("--ref_image", default="white", choices=["self", "white", "black", "noise"], help="Reference image for calibration")
parser.add_argument("--beta", type=float, default=0, help="Beta parameter for calibration")
parser.add_argument("--calibration_query", default="nana pina sequ zuta rupi", help="Query for calibration")
parser.add_argument("--input_token_idx_calibration", type=int, nargs="+", default=[-1, -2, -3], help="Token indices for calibration")

parser.add_argument("--do_sample", action="store_true", default=False, help="Use sampling for generation")
parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for generation")
parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument("--load_in_8bit", action="store_true", default=True, help="Load model in 8-bit precision")

args = parser.parse_args()


def dynamic_generate(raw_image, query, exp_config, model, processor, model_type):
    """
    Custom generation loop with dynamic attention scaling.
    """
    try:
        inputs = process_inputs(raw_image, query, processor, model_type)
        generated = inputs["input_ids"]
        input_length = inputs["input_ids"].shape[-1]
        img_token_idxs = torch.nonzero(inputs['input_ids'][0] == model.config.image_token_index, as_tuple=False).flatten().cpu()
        modifier = SelfAttentionModifier(model, exp_config, img_token_idxs)
        attention_maps = {}
        
        if exp_config['ref_image'] == 'self':
            input_image = raw_image
        elif exp_config['ref_image'] == 'white':
            input_image = Image.new("RGB", (224, 224), (255, 255, 255))
        elif exp_config['ref_image'] == 'black':
            input_image = Image.new("RGB", (224, 224), (0, 0, 0))
        elif exp_config['ref_image'] == 'noise':
            input_image = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8), mode="RGB")
        
        modifier.update_calibration_matrix(attention_maps, input_image, processor, model_type)
        modifier.register_hooks()
        
        with torch.no_grad():
            for _ in range(exp_config['max_new_tokens']):
                inputs["input_ids"] = generated
                batch_size, seq_len = generated.shape
                inputs["attention_mask"] = torch.ones((batch_size, seq_len), device=generated.device, dtype=torch.long)
                outputs = model(**inputs, output_attentions=True)
                last_logits = outputs.logits[:, -1, :]
                confidence = torch.max(torch.nn.functional.softmax(last_logits, dim=-1)).item()
                
                if confidence > exp_config['confidence_threshold']:
                    modifier.dynamic_factor = exp_config['compute_attention_factor'](exp_config, 1)
                else:
                    modifier.dynamic_factor = exp_config['compute_attention_factor'](exp_config, confidence)
                
                next_token = model.generate(**inputs, do_sample=args.do_sample, max_new_tokens=1, num_beams=args.num_beams)[:, -1:]
                generated = torch.cat([generated, next_token], dim=-1)
                
                if next_token.item() == processor.tokenizer.eos_token_id:
                    break
                
                del outputs, last_logits, confidence, next_token
                torch.cuda.empty_cache()
        
        modifier.remove_hooks()
        return generated[:, input_length:]
    except Exception as e:
        logging.error(f"Error in dynamic_generate: {e}")
        raise


def load_opera_image_ids(opera_results_path):
    image_ids = []
    try:
        with open(opera_results_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                image_ids.append(data['image_id'])
        return image_ids
    except Exception as e:
        logging.error(f"Error reading OPERA results file {opera_results_path}: {e}")
        raise

def main():
    """
    Main function to evaluate the model on the CHAIR benchmark.
    """
    try:
        # Set up device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Create log directory
        model_log_dir = os.path.join(args.log_dir, args.model_type)
        os.makedirs(model_log_dir, exist_ok=True)
        log_file = os.path.join(model_log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        # Load model and processor
        model_names = {
            "instructblip": "Salesforce/instructblip-vicuna-7b",
            "llava": "llava-hf/llava-1.5-7b-hf"
        }
        logger.info(f"Loading model: {args.model_type}")
        model, processor = load_model_and_processor(args.model_type, model_names, args.cache_dir, device, args.load_in_8bit)
        model.eval()
        logger.info("Model and processor loaded successfully")
        logger.info(f"OPERA results path: {args.opera_results}")

        # Set up experiment configuration
        exp_config = {
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
        exp_id = np.random.randint(1000, 9999)
        config_path = os.path.join(model_log_dir, f"config_{exp_id}.json")
        config_to_save = {k: v for k, v in exp_config.items() if k != "compute_attention_factor"}
        with open(config_path, 'w') as f:
            json.dump(config_to_save, f, indent=4)
        logger.info(f"Saved experiment configuration to {config_path}")

        # Set up output file
        output_file = os.path.join(model_log_dir, f"{args.model_type}_{exp_id}.jsonl")
        logger.info(f"Output will be saved to {output_file}")

        # List all images in chair_path
        if args.opera_results:
            image_ids = load_opera_image_ids(args.opera_results)
            image_list = [(f"COCO_val2014_{id:012d}.jpg", id) for id in image_ids]
            logger.info(f"Loaded {len(image_list)} image IDs from OPERA results")
        else:
            raise NotImplementedError("Failed to load images from OPERA results")
            img_files = [f for f in os.listdir(args.chair_path) if f.endswith('.jpg')]
            image_list = [(f, int(f.split(".jpg")[0][-6:])) for f in img_files]
            random.seed(args.random_seed)
            random.shuffle(image_list)
            if args.num_images is not None:
                image_list = image_list[:args.num_images]
            logger.info(f"Processing {len(image_list)} images from {args.chair_path}")

        # Process each image
        for img_file, img_id in tqdm(image_list, desc="Generating captions"):
            img_path = os.path.join(args.chair_path, img_file)
            if not os.path.exists(img_path):
                logger.warning(f"Image {img_path} not found, skipping")
                continue
            try:
                raw_image = Image.open(img_path).convert("RGB")
                query = "Please describe this image in detail."
                generated_ids = dynamic_generate(raw_image, query, exp_config, model, processor, args.model_type)
                caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                img_save = {
                    "image_id": img_id,
                    "caption": caption
                }
                with open(output_file, "a") as f:
                    json.dump(img_save, f)
                    f.write('\n')
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                logger.error(f"Error processing image {img_file}: {e}")
                continue

        logger.info(f"Evaluation complete. Results saved to {output_file}")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()