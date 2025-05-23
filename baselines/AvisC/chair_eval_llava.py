import argparse
import torch
import os
import json
from tqdm import tqdm
from PIL import Image
import random
import numpy as np
import logging
from datetime import datetime
import gc
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/experiments')

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from avisc_utils.vcd_add_noise import add_diffusion_noise
from avisc_utils.avisc_sample import evolve_avisc_sampling
from utils import dist_util  # Assuming this utility is available

evolve_avisc_sampling()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLaVA on CHAIR with hallucination mitigation")
    parser.add_argument("--model-path", type=str, default="/path/to/llava-v1.5-7b/", help="Path to LLaVA model")
    parser.add_argument("--model-base", type=str, default=None, help="Base model path, if applicable")
    
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--opera_results", type=str, default='/path/to/OPERA/llava/ours.jsonl', help="Path to OPERA results JSONL file for image IDs") ################
    parser.add_argument("--log_dir", type=str, default="./results/VCD", help="Directory for logs and results") ################
    parser.add_argument("--data_path", type=str, default="/path/to/coco2014/val2014/", help="Path to COCO2014 val images")
    parser.add_argument("--num_images", type=int, default=500, help="Number of images to process when not using opera_results")

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", type=str2bool, default=True)
    parser.add_argument("--cd_alpha", type=float, default=1.0)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)    

    parser.add_argument("--use_avisc", type=str2bool, default=False)
    parser.add_argument("--layer_gamma", type=float, default=0.1)
    parser.add_argument("--masking_scheme", type=str, default="zeros")
    parser.add_argument("--lamb", type=float, default=1.0)
    parser.add_argument("--exp_description", type=str, default="..")
    parser.add_argument("--max_token", type=int, default=64)
    parser.add_argument("--use_m3id", type=str2bool, default=False)
    
    return parser.parse_args()

def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

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
    args = parse_args()

    # Setup DDP
    dist_util.setup_dist(args)
    device = dist_util.device()

    # Set up logging
    model_log_dir = os.path.join(args.log_dir, "llava")
    if dist.get_rank() == 0:
        os.makedirs(model_log_dir, exist_ok=True)
        log_file = os.path.join(model_log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {device}")
    logger.info(f"use_cd: {args.use_cd}, use_avisc: {args.use_avisc}, cd_alpha: {args.cd_alpha}, layer_gamma: {args.layer_gamma}, masking_scheme: {args.masking_scheme}, lamb: {args.lamb}")

    # Save arguments to config file
    exp_id = np.random.randint(1000, 9999)
    config_path = os.path.join(model_log_dir, f"{exp_id}_config.json")
    if dist.get_rank() == 0:
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=4)
        logger.info(f"Saved configuration to {config_path}")

    # Set random seed
    setup_seeds(args.seed)

    # Load LLaVA model, tokenizer, and image processor
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    tokenizer.padding_side = "left"
    model.to(device)
    logger.info("Model, tokenizer, and image processor loaded successfully")

    # Load CHAIR image list
    if args.opera_results:
        image_ids = load_opera_image_ids(args.opera_results)
        image_list = [(f"COCO_val2014_{id:012d}.jpg", id) for id in image_ids]
        logger.info(f"Loaded {len(image_list)} image IDs from OPERA results")
    else:
        img_files = [f for f in os.listdir(args.data_path) if f.endswith('.jpg')]
        image_list = [(f, int(f.split(".jpg")[0][-6:])) for f in img_files]
        random.shuffle(image_list)
        if args.num_images is not None:
            image_list = image_list[:args.num_images]
        logger.info(f"Processing {len(image_list)} images from {args.data_path}")

    # Setup output file
    output_file = os.path.join(model_log_dir, f"{exp_id}_llava_results.jsonl")
    logger.info(f"Output will be saved to {output_file}")

    # Process each image
    for img_file, img_id in tqdm(image_list, desc="Generating captions"):
        img_path = os.path.join(args.data_path, img_file)
        if not os.path.exists(img_path):
            logger.warning(f"Image {img_path} not found, skipping")
            continue
        try:
            raw_image = Image.open(img_path).convert("RGB")
            query = "Please describe this image in detail."

            # Prepare prompt
            if model.config.mm_use_im_start_end:
                qu = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + query
            else:
                qu = DEFAULT_IMAGE_TOKEN + '\n' + query
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qu)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # Tokenize prompt
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

            # Process image
            image_tensor = image_processor(raw_image, return_tensors="pt")['pixel_values'].to(device).half()

            # Prepare contrastive decoding image if enabled
            if args.use_cd:
                image_tensor_cd = add_diffusion_noise(image_tensor, noise_step=args.noise_step)
            else:
                image_tensor_cd = None

            # Set up stopping criteria
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            # Generate caption with mitigation methods
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    images_cd=image_tensor_cd,
                    cd_alpha=args.cd_alpha,
                    cd_beta=args.cd_beta,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=args.max_token,
                    use_cache=True,
                    use_avisc=args.use_avisc,
                    layer_gamma=args.layer_gamma,
                    masking_scheme=args.masking_scheme,
                    lamb=args.lamb,
                    use_m3id=args.use_m3id,
                    # stopping_criteria=[stopping_criteria]
                )

            # Decode output
            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
            caption = outputs[0].strip()
            if caption.endswith(stop_str):
                caption = caption[:-len(stop_str)].strip()

            # Compute token length
            token_ids = tokenizer(caption, add_special_tokens=False)["input_ids"]
            token_len = len(token_ids)

            # Save result
            result = {
                "image_id": img_id,
                "caption": caption,
                "length": token_len
            }
            if dist.get_rank() == 0:
                with open(output_file, "a") as f:
                    json.dump(result, f)
                    f.write('\n')

            # Log query and caption
            logger.info(f"[{img_file}]")
            logger.info(f"Q: {query}")
            logger.info(f"A: {caption}")

            # Clear memory
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            logger.error(f"Error processing image {img_file}: {e}")
            continue

    logger.info(f"Evaluation complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()