import os
import gc
import json
import torch
from PIL import Image
from tqdm import tqdm
import argparse
import logging
from datetime import datetime
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/src')

from model_utils import load_model_and_processor, process_inputs
from CAAC_utils import SelfAttentionModifier, compute_attention_factor



# Set up argument parsing
parser = argparse.ArgumentParser(description="Run POPE benchmark with hallucination mitigation on all strategies")
parser.add_argument("--model_type", default="instructblip", choices=["instructblip", "llava"], help="Model type")
parser.add_argument("--load_in_8bit", action="store_true", default=True, help="Load model in 8-bit precision")
parser.add_argument("--cache_dir", default="/path/to/model_checkpoints", help="Cache directory for model")
parser.add_argument("--question_dir", default="/path/to/POPE/coco", help="Directory containing POPE question JSON files")
parser.add_argument("--image_folder", default="/path/to/POPE/coco/val2014", help="Path to image folder")
parser.add_argument("--log_dir", default="../results/POPE", help="Path to amber directory")

parser.add_argument("--do_sample", action="store_true", default=False, help="Use sampling for generation")
parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for generation")
parser.add_argument("--max_new_tokens", type=int, default=10, help="Maximum new tokens to generate")
parser.add_argument("--img_txt_cal_layers", type=int, nargs="+", default=list(range(32)), help="Layers for upscaling intervention")
parser.add_argument("--img_cal_layers", type=int, nargs="+", default=[0, 1], help="Layers for calibration intervention")
parser.add_argument("--min_lamb", type=float, default=1.0, help="Minimum lambda for attention scaling")
parser.add_argument("--max_lamb", type=float, default=1.5, help="Maximum lambda for attention scaling")
parser.add_argument("--confidence_threshold", type=float, default=0.25, help="Confidence threshold for dynamic scaling")
parser.add_argument("--calibration_query", default="# # # # # # # # # #", help="Query for calibration")
parser.add_argument("--input_token_idx_calibration", type=int, nargs="+", default=[-1, -2, -3], help="Token indices for calibration")
parser.add_argument("--ref_image", default="white", choices=["self", "white", "black", "noise"], help="Reference image for calibration")
parser.add_argument("--beta", type=float, default=0.5, help="Beta parameter for calibration")
args = parser.parse_args()

# Define all prompting strategies
STRATEGIES = ["adversarial", "popular", "random"]

# Set up logging
# log_dir = os.path.join(args.results_dir, args.model_type)
log_dir = args.log_dir
os.makedirs(log_dir, exist_ok=True)
LOG_FILE = os.path.join(log_dir, f"log_all_strategies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model names
model_names = {
    "instructblip": "Salesforce/instructblip-vicuna-7b",
    "llava": "llava-hf/llava-1.5-7b-hf"
}

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Experiment configuration
EXP_ID = np.random.randint(1000, 9999)
EXP_CONFIG_PATH = os.path.join(log_dir, f"{args.model_type}_{EXP_ID}_config.json")
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

# Save EXP_CONFIG to JSON
config_to_save = {k: v for k, v in EXP_CONFIG.items() if k != "compute_attention_factor"}
with open(EXP_CONFIG_PATH, 'w') as file:
    json.dump(config_to_save, file, indent=4)
logger.info(f"Saved experiment configuration to {EXP_CONFIG_PATH}")

def dynamic_generate(raw_image, query, exp_config, model, processor, model_type):
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

def process_strategy(strategy, model, processor, args):
    try:
        # Derive question file and answers path
        question_file = os.path.join(args.question_dir, f"coco_pope_{strategy}.json")
        answers_path = os.path.join(log_dir, f"{strategy}.json")
        
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
                try:
                    raw_image = Image.open(img_path).convert('RGB')
                except Exception as e:
                    logger.error(f"Error loading image {img_path}: {e}")
                    continue
                
                generated_ids = dynamic_generate(raw_image, prompt, EXP_CONFIG, model, processor, args.model_type)
                response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                ans_file.write(json.dumps({
                    "question_id": question_id,
                    "prompt": prompt,
                    "text": response_text,
                    "model_id": args.model_type,
                    "image": image_file,
                    "metadata": {}
                }) + "\n")
                ans_file.flush()
                
                torch.cuda.empty_cache()
                gc.collect()
        logger.info(f"Saved answers for {strategy} to {answers_path}")
    except Exception as e:
        logger.error(f"Error processing strategy {strategy}: {e}")
        raise

def main():
    try:
        # Load model and processor
        model, processor = load_model_and_processor(args.model_type, model_names, args.cache_dir, device, args.load_in_8bit)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Process each strategy
        for strategy in STRATEGIES:
            process_strategy(strategy, model, processor, args)
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()