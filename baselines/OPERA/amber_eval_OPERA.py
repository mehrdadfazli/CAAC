import os
import json
import random
import gc
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from PIL import Image

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.models import load_preprocess
from torchvision.transforms import ToTensor, Resize, Compose



# Mapping for configuration files (adjust as needed)
MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "instructblip": "eval_configs/instructblip_eval.yaml",
    "lrv_instruct": "eval_configs/lrv_instruct_eval.yaml",
    "shikra": "eval_configs/shikra_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:"
}

AMBER_PATH = {
    "query": "/projects/zzhu20/Mehrdad/AMBER/data/query/query_all.json",
    "annotations": "/projects/zzhu20/Mehrdad/AMBER/data/annotations.json",
    "images": "/projects/zzhu20/Mehrdad/AMBER/image/",
}

# =============================================================================
# Set parameters manually (adjust these values for your setup)
# =============================================================================
model_name = "instructblip"  # change to your model name if needed
use_opera = False
query_json = AMBER_PATH["query"]  # update with the path to your AMBER query JSON file
anno_json = AMBER_PATH["annotations"]  # optional: path to AMBER annotation JSON if available
image_dir = AMBER_PATH["images"]  # directory containing the images
gpu_id = 0
num_beams = 1
scale_factor = 50
threshold = 15
num_attn_candidates = 5
penalty_weights = 1.0
max_new_tokens = 64
output_path = f"log/AMBER/{model_name}_noOPERA_amber_{max_new_tokens}_all.json"  # where to save the output
config_file_path = f"log/AMBER/{model_name}_noOPERA_amber_{max_new_tokens}_all_config.json"  # Choose your desired output file name

image_size = 336 if model_name=="llava-1.5" else 224
transform = Compose([
    Resize((image_size, image_size)),  # Optional: Resize the image if required by your model
    ToTensor(),          # Converts PIL Image to torch.Tensor (scales pixel values to [0, 1])
])
# =============================================================================
# Create a simple args-like object to mimic command-line arguments
# =============================================================================
class Args:
    pass

args = Args()
args.model = model_name
args.use_opera = use_opera
args.query_json = query_json
args.anno_json = anno_json
args.image_dir = image_dir
args.gpu_id = gpu_id
args.num_beams = num_beams
args.scale_factor = scale_factor
args.threshold = threshold
args.num_attn_candidates = num_attn_candidates
args.penalty_weights = penalty_weights
args.max_new_tokens = max_new_tokens
args.output_path = output_path
args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]

args.options = []


# =============================================================================
# Create a dictionary to hold the experiment configurations
# =============================================================================
experiment_config = {
    "model_name": model_name,
    "use_opera": use_opera,
    "query_json": query_json,
    "anno_json": anno_json,
    "image_dir": image_dir,
    "gpu_id": gpu_id,
    "num_beams": num_beams,
    "scale_factor": scale_factor,
    "threshold": threshold,
    "num_attn_candidates": num_attn_candidates,
    "penalty_weights": penalty_weights,
    "max_new_tokens": max_new_tokens,
    "output_path": output_path, 
    # Note: We are not saving the 'args' object directly as it might contain
    # methods or complex objects that are not easily serializable to JSON.
    # We are explicitly including the attributes we need.
}

# =============================================================================
# Save the dictionary to a JSON file
# =============================================================================


with open(config_file_path, 'w') as f:
    json.dump(experiment_config, f, indent=4)
    

# =============================================================================
# Utility functions
# =============================================================================
def recorder(out):
    NEG_WORDS = ["No", "not", "no", "NO"]

    out = out.replace('.', '')
    out = out.replace(',', '')
    words = out.split(' ')
    if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
        return "No"
    else:
        return "Yes"
    
def setup_seeds(cfg):
    seed = cfg.run_cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def process_inputs(raw_image, query, vis_processor):
    """
    Process the raw image and query using the provided visual processor.
    Returns a dictionary formatted for model.generate.
    """
    image = vis_processor(raw_image)
    return {"image": image, "prompt": query}

# =============================================================================
# Main evaluation function for AMBER
# =============================================================================
def run_amber_evaluation(args):
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # Load configuration from the YAML file
    cfg = Config(args)
    setup_seeds(cfg)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Initialize the model
    print("Initializing Model...")
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)


    model.eval()

    # Load image and text processors
    vis_processors, txt_processors = load_preprocess(cfg.get_config().preprocess)
    vis_processor = vis_processors["eval"]
    txt_processor = txt_processors["eval"] if "eval" in txt_processors else None

    # Load AMBER query data
    with open(args.query_json, 'r') as f:
        data = json.load(f)

    # Optionally load annotations (if available; not used for decoding here)
    annotations = None
    if args.anno_json is not None:
        with open(args.anno_json, 'r') as f:
            annotations = json.load(f)

    responses = []
    print("Starting AMBER evaluation with OPERA decoding...")

    for item in tqdm(data, desc="Processing AMBER Dataset"):
        
        if item['id'] > 1004:
            args.max_new_tokens = 20


        image_id = item['id']
        image_file = item['image']
        img_path = os.path.join(args.image_dir, image_file)

        raw_image = Image.open(img_path).convert('RGB')
        img_tensor = transform(raw_image).unsqueeze(0)
        img_tensor = img_tensor.to(device)

        query = item['query']
        template = INSTRUCTION_TEMPLATE[args.model]
        qu = template.replace("<question>", query)

        with torch.inference_mode():
            with torch.no_grad():
                out = model.generate(
                    {"image": img_tensor, "prompt":qu}, 
                    use_nucleus_sampling=True, 
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    opera_decoding=args.use_opera,
                    scale_factor=args.scale_factor,
                    threshold=args.threshold,
                    num_attn_candidates=args.num_attn_candidates,
                    penalty_weights=args.penalty_weights,
                )
            
    
        if item['id'] > 1004:
            out[0] = recorder(out[0])
        
        responses.append({'id': image_id, 'response': out[0]})
    
        # Clean up to free memory
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save responses to a JSON file
    with open(args.output_path, 'w') as f:
        json.dump(responses, f, indent=4)
    print(f"Evaluation completed. Results saved to {args.output_path}")
    
    return responses

# =============================================================================
# Run the evaluation and inspect some outputs
# =============================================================================
responses = run_amber_evaluation(args)
print("First few responses:")
for r in responses[:5]:
    print(r)