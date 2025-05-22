import os
import gc
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import logging
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/src')

from model_utils import load_model_and_processor, process_inputs
from CAAC_utils import SelfAttentionModifier, compute_attention_factor

# Set up logging
parser = argparse.ArgumentParser(description="Run AMBER benchmark with hallucination mitigation")
parser.add_argument("--model_type", default="instructblip", choices=["instructblip", "llava"], help="Model type")
parser.add_argument("--load_in_8bit", action="store_true", default=True, help="Load model in 8-bit precision")
parser.add_argument("--cache_dir", default="/path/to/model_checkpoints", help="Cache directory for model")
parser.add_argument("--amber_path", default="/projects/zzhu20/Mehrdad/AMBER", help="Path to AMBER dataset")
parser.add_argument("--log_dir", default="../results/AMBER", help="Directory for logs and results")
parser.add_argument("--use_CAAC", action="store_true", default=True, help="Use CAAC for hallucination mitigation")

parser.add_argument("--img_txt_cal_layers", type=int, nargs="+", default=list(range(32)), help="Layers for upscaling intervention")
parser.add_argument("--img_cal_layers", type=int, nargs="+", default=list(range(10)), help="Layers for calibration intervention")
parser.add_argument("--min_lamb", type=float, default=1.0, help="Minimum lambda for attention scaling")
parser.add_argument("--max_lamb", type=float, default=1.5, help="Maximum lambda for attention scaling")
parser.add_argument("--confidence_threshold", type=float, default=0.25, help="Confidence threshold for dynamic scaling")
parser.add_argument("--ref_image", default="white", choices=["self", "white", "black", "noise"], help="Reference image for calibration")
parser.add_argument("--beta", type=float, default=0, help="Beta parameter for calibration")
parser.add_argument("--calibration_query", default="# # # # # # # # # #", help="Query for calibration")
parser.add_argument("--input_token_idx_calibration", type=int, nargs="+", default=[-1, -2, -3], help="Token indices for calibration")

parser.add_argument("--do_sample", action="store_true", default=False, help="Use sampling for generation")
parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for generation")
parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")

args = parser.parse_args()

os.makedirs(args.log_dir, exist_ok=True)
LOG_FILE = os.path.join(args.log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Derived paths
EXP_ID = np.random.randint(1000,9999)
JSON_QUERY_PATH = os.path.join(args.amber_path, "data/query/query_all.json")
JSON_ANNOTATION_PATH = os.path.join(args.amber_path, "data/annotations.json")
IMAGE_DIR = os.path.join(args.amber_path, "image")
EXP_CONFIG_PATH = os.path.join(args.log_dir, f"amber_{args.model_type}_{EXP_ID}_config.json")
RESPONSES_PATH = os.path.join(args.log_dir, f"amber_{args.model_type}_{EXP_ID}_response.json")

# Model names
model_names = {
    "instructblip": "Salesforce/instructblip-vicuna-7b",
    "llava": "llava-hf/llava-1.5-7b-hf"
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

def recorder(out):
    NEG_WORDS = ["No", "not", "no", "NO"]

    out = out.replace('.', '')
    out = out.replace(',', '')
    words = out.split(' ')
    if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
        return "No"
    else:
        return "Yes"

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
        logger.error(f"Error in dynamic_generate: {e}")
        raise

def main():
    """
    Main function to run the AMBER benchmark.
    """
    try:
        # Load model and processor
        model, processor = load_model_and_processor(args.model_type, model_names, args.cache_dir, device, args.load_in_8bit)
        tokenizer = processor.tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Load dataset and annotations
        with open(JSON_QUERY_PATH, 'r') as file:
            data = json.load(file)
        with open(JSON_ANNOTATION_PATH, 'r') as file:
            annotations = json.load(file)
        logger.info("Loaded dataset and annotations")
        
        # Load or initialize responses
        responses = []
        if os.path.exists(RESPONSES_PATH):
            try:
                with open(RESPONSES_PATH, 'r') as file:
                    responses = json.load(file)
                logger.info(f"Loaded existing responses from {RESPONSES_PATH}")
            except Exception as e:
                logger.error(f"Error reading responses file: {e}")
        
        processed_ids = set(item['id'] for item in responses)
        
        # Process dataset
        for item in tqdm(data, desc="Processing Dataset"):
            if item['id'] > 1004: #- reducing the number of max tokens for discriminative tasks
                EXP_CONFIG["max_new_tokens"] = 10

            if item['id'] in processed_ids:
                continue
            image_id = item['id']
            image_file = item['image']
            img_path = os.path.join(IMAGE_DIR, image_file)
            try:
                raw_image = Image.open(img_path).convert('RGB')
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
                continue
            query = item['query']

            if args.use_CAAC:
                generated_ids = dynamic_generate(raw_image, query, EXP_CONFIG, model, processor, args.model_type)
                
                if item['id'] <= 1004:
                    response_text_64 = processor.batch_decode(generated_ids[..., :64], skip_special_tokens=True)[0].strip()
                    # response_text_128 = processor.batch_decode(generated_ids[..., :128], skip_special_tokens=True)[0].strip()
                    # response_text_256 = processor.batch_decode(generated_ids[..., :256], skip_special_tokens=True)[0].strip()
                    response_text_512 = processor.batch_decode(generated_ids[..., :512], skip_special_tokens=True)[0].strip()
                    

                    #- recordning the responses of various lengths
                    responses.append({
                        'id': image_id,
                        'response_64': response_text_64,
                        # 'response_128': response_text_128,
                        # 'response_256': response_text_256,
                        'response_512': response_text_512,
                        'response_length': generated_ids.shape[-1]
                    })
                else:
                    response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                    response_text = recorder(response_text)

                    responses.append({
                        'id': image_id,
                        'response': response_text,
                        'response_length': generated_ids.shape[-1]
                    })
            else:
                inputs = process_inputs(raw_image, query, processor, args.model_type)

                generated_ids = model.generate(
                    **inputs,
                    do_sample= args.do_sample,
                    max_new_tokens= args.max_new_tokens,
                    num_beams=args.num_beams
                )
                if item['id'] <= 1004:
                    response_text_64 = processor.batch_decode(generated_ids[..., :64], skip_special_tokens=True)[0].strip()
                    # response_text_128 = processor.batch_decode(generated_ids[..., :128], skip_special_tokens=True)[0].strip()
                    # response_text_256 = processor.batch_decode(generated_ids[..., :256], skip_special_tokens=True)[0].strip()
                    response_text_512 = processor.batch_decode(generated_ids[..., :512], skip_special_tokens=True)[0].strip()

                    responses.append({
                        'id': image_id,
                        'response_64': response_text_64,
                        # 'response_128': response_text_128,
                        # 'response_256': response_text_256,
                        'response_512': response_text_512,
                        'response_length': generated_ids.shape[-1]
                    })
                else:
                    response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                    response_text = recorder(response_text)

                    responses.append({
                        'id': image_id,
                        'response': response_text,
                        'response_length': generated_ids.shape[-1]
                    })
                    
            processed_ids.add(image_id)
            try:
                with open(RESPONSES_PATH, 'w') as file:
                    json.dump(responses, file, indent=4)
                logger.info(f"Saved responses to {RESPONSES_PATH}")
            except Exception as e:
                logger.error(f"Error saving responses: {e}")
            torch.cuda.empty_cache()
            gc.collect()
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()