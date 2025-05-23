import argparse
import torch
import os
import json
from tqdm import tqdm
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import logging
from datetime import datetime
from PIL import Image
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/experiments')


from llava.utils import disable_torch_init
from lavis.models import load_model_and_preprocess
from avisc_utils.vcd_add_noise import add_diffusion_noise
from avisc_utils.avisc_sample import evolve_avisc_sampling
from utils import dist_util
from utils.logger import create_logger

evolve_avisc_sampling()
torch.multiprocessing.set_sharing_strategy('file_system')

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
    parser = argparse.ArgumentParser(description="CHAIR evaluation on InstructBLIP with hallucination mitigation - VCD")
    parser.add_argument("--model-path", type=str, default="path/checkpoints/instruct_blip", help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, default="/path/to/coco2014/val2014/", help="Path to COCO2014 val images")
    parser.add_argument("--log_dir", type=str, default="./results/VCD", help="Directory for logs and results") ################
    parser.add_argument("--opera_results", type=str, default='/path/to/OPERA/instructblip/ours.jsonl', help="Path to OPERA results JSONL file for image IDs") ################
    parser.add_argument("--num_images", type=int, default=500, help="Number of images to process when not using opera_results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of data loader workers")
    parser.add_argument("--use_avisc", type=str2bool, default=False, help="Use AVISC mitigation") ################
    parser.add_argument("--layer_gamma", type=float, default=0.1, help="Layer gamma for AVISC") ################
    parser.add_argument("--masking_scheme", type=str, default="zeros", help="Masking scheme for AVISC")
    parser.add_argument("--lamb", type=float, default=1, help="Lambda for AVISC")
    parser.add_argument("--use_cd", type=str2bool, default=True, help="Use contrastive decoding") ################
    parser.add_argument("--cd_alpha", type=float, default=1.0, help="CD alpha parameter") ################
    parser.add_argument("--cd_beta", type=float, default=0.1, help="CD beta parameter")
    parser.add_argument("--use_m3id", type=str2bool, default=False, help="Use M3ID mitigation") ################
    parser.add_argument("--noise_step", type=int, default=500, help="Noise step for contrastive decoding")
    parser.add_argument("--max_token", type=int, default=64, help="Maximum tokens to generate")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling parameter")
    args = parser.parse_args()
    return args

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

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Setup DDP
    dist_util.setup_dist(args)
    device = dist_util.device()

    # Setup experiment folder
    model_log_dir = os.path.join(args.log_dir, "instructblip")
    if dist.get_rank() == 0:
        os.makedirs(model_log_dir, exist_ok=True)
        log_file = os.path.join(model_log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        logger.info(f"Experiment directory created at {model_log_dir}")
        logger.info(f"use_cd: {args.use_cd}, use_avisc: {args.use_avisc}, cd_alpha: {args.cd_alpha},layer_gamma: {args.layer_gamma}, masking_scheme: {args.masking_scheme}, lamb: {args.lamb}")

    # Initialize model
    disable_torch_init()
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
    tokenizer = model.llm_tokenizer
    logger.info("Model and processor loaded successfully")

    # Set random seed
    setup_seeds(args.seed)

    # Load image list
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
    exp_id = np.random.randint(1000, 9999)
    output_file = os.path.join(model_log_dir, f"{exp_id}_instructblip_results.jsonl")
    logger.info(f"Output will be saved to {output_file}")

    # Process images
    for img_file, img_id in tqdm(image_list, desc="Generating captions"):
        img_path = os.path.join(args.data_path, img_file)
        if not os.path.exists(img_path):
            logger.warning(f"Image {img_path} not found, skipping")
            continue
        try:
            # Load and preprocess image
            raw_image = Image.open(img_path).convert("RGB")
            image = vis_processors["eval"](raw_image).unsqueeze(0)

            # Prepare query
            query = "Please describe this image in detail."

            # Apply contrastive decoding noise if enabled
            if args.use_cd:
                image_tensor_cd = add_diffusion_noise(image, args.noise_step)
            else:
                image_tensor_cd = None

            # Generate caption
            with torch.inference_mode():
                output = model.generate(
                    {"image": image.to(device), "prompt": query},
                    use_nucleus_sampling=True,
                    num_beams=1,
                    top_p=args.top_p,
                    repetition_penalty=1,
                    images_cd=image_tensor_cd.half().to(device) if image_tensor_cd is not None else None,
                    cd_beta=args.cd_beta,
                    use_avisc=args.use_avisc,
                    layer_gamma=args.layer_gamma,
                    masking_scheme=args.masking_scheme,
                    lamb=args.lamb,
                    max_length=args.max_token,
                    cd_alpha=args.cd_alpha,
                    use_m3id=args.use_m3id,
                )[0]

            # Compute token length
            token_ids = tokenizer(output, add_special_tokens=False)["input_ids"]
            token_len = len(token_ids)

            # Save result
            result = {
                "image_id": img_id,
                "caption": output,
                "length": token_len
            }
            with open(output_file, "a") as f:
                json.dump(result, f)
                f.write('\n')

            logger.info(f"[{img_file}]")
            logger.info(f"Q: {query}")
            logger.info(f"A: {output}")

            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error processing image {img_file}: {e}")
            continue

    logger.info(f"Evaluation complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()