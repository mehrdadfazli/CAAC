import torch
from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
    AutoProcessor
)
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

def load_model_and_processor(model_type, model_names, cache_dir, device, load_in_8bit=True):
    """
    Load the model and processor based on the model type.
    """
    try:
        quantization_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit, llm_int8_threshold=200.0)
        if model_type == "instructblip":
            model = InstructBlipForConditionalGeneration.from_pretrained(
                model_names[model_type],
                torch_dtype=torch.float16,
                attn_implementation="eager",
                cache_dir=cache_dir,
                quantization_config=quantization_config,
                device_map="auto"
            )
            model.tie_weights()
            processor = InstructBlipProcessor.from_pretrained(model_names[model_type], cache_dir=cache_dir)
        elif model_type == "llava":
            model = LlavaForConditionalGeneration.from_pretrained(
                model_names[model_type],
                torch_dtype=torch.float16,
                attn_implementation="eager",
                cache_dir=cache_dir,
                quantization_config=quantization_config,
                device_map="auto"
            )
            processor = AutoProcessor.from_pretrained(model_names[model_type], cache_dir=cache_dir)
            processor.patch_size = model.config.vision_config.patch_size
            processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        logger.info(f"Loaded {model_type} model and processor")
        return model, processor
    except Exception as e:
        logger.error(f"Error loading model and processor: {e}")
        raise

def process_inputs(raw_image, query, processor, model_type, device="cuda"):
    """
    Process inputs depending on the model (e.g., InstructBLIP or LLaVA).
    """
    try:
        if model_type == "llava":
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": query},
                    ],
                }
            ]
            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=raw_image, text=text_prompt, padding=True, return_tensors="pt").to(device, torch.float16)
        elif model_type == "instructblip":
            inputs = processor(images=raw_image, text=query, return_tensors="pt").to(device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        return inputs
    except Exception as e:
        logger.error(f"Error processing inputs: {e}")
        raise

def get_token_probability(model, inputs, token_id):
    """
    Compute the probability of a specific token given model inputs.
    """
    try:
        outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits.detach(), dim=-1)
        token_prob = probabilities[0, -1, token_id].item()
        return token_prob
    except Exception as e:
        logger.error(f"Error computing token probability: {e}")
        raise
    