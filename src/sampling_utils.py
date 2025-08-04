import torch
import numpy as np
import math
import gc
import torch.nn.functional as F

def add_diffusion_noise(image_tensor, noise_step):
    num_steps = 1000
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0, t):
        noise = torch.randn_like(x_0)
        return alphas_bar_sqrt[t] * x_0 + one_minus_alphas_bar_sqrt[t] * noise

    noise_step = int(noise_step)
    return q_x(image_tensor.clone(), noise_step)
    
def generate_VCD(
    model, tokenizer, inputs,
    max_new_tokens=128, do_sample=False, raw_image=None, return_n_tokens=False,
    vcd_alpha=1.0, vcd_beta=0.1, vcd_noise_step=500
):
    """
    Implements token-level generation with Visual Contrastive Decoding (VCD) for HuggingFace LLaVA-NeXT.
    Uses separate past_key_values for clean and distorted images to avoid redundant recomputation.
    """
    model.eval()
    generated_tokens = []

    input_ids = inputs["input_ids"]  # (1, T)
    pixel_values = inputs["pixel_values"]  # (1, 3, H, W)
    attention_mask = inputs.get("attention_mask", None)
    image_sizes = inputs.get("image_sizes", None)

    if image_sizes is None:
        assert raw_image is not None
        image_sizes = [(raw_image.height, raw_image.width)]

    # === Generate distorted version ===
    with torch.no_grad():
        distorted_pixel_values = add_diffusion_noise(pixel_values, noise_step=vcd_noise_step).clamp(0, 1)

    past_key_values = None
    past_key_values_cd = None
    next_token_id = None

    for step in range(max_new_tokens):
        is_first_step = step == 0
        current_input_ids = input_ids if is_first_step else next_token_id  # (1, 1)

        # === Clean forward ===
        with torch.no_grad():
            outputs = model(
                input_ids=current_input_ids,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False
            )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

        # === Distorted forward ===
        with torch.no_grad():
            outputs_cd = model(
                input_ids=current_input_ids,
                pixel_values=distorted_pixel_values,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
                past_key_values=past_key_values_cd,
                use_cache=True,
                output_attentions=False
            )
            logits_cd = outputs_cd.logits[:, -1, :]
            past_key_values_cd = outputs_cd.past_key_values

        # === VCD contrastive decoding ===
        cutoff = torch.log(torch.tensor(vcd_beta).to(logits.device)) + logits.max(dim=-1, keepdim=True).values
        adjusted_logits = (1 + vcd_alpha) * logits - vcd_alpha * logits_cd
        adjusted_logits = adjusted_logits.masked_fill(logits < cutoff, -float("inf"))

        # === Token selection ===
        if do_sample:
            raise NotImplementedError("'do_sample=True' is not implemented.")
        else:
            next_token_id = torch.argmax(adjusted_logits, dim=-1, keepdim=True)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

        generated_tokens.append(next_token_id)

        # === Update attention mask and input_ids ===
        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)
            ], dim=-1)

    if not generated_tokens:
        return ""

    generated_sequence = torch.cat(generated_tokens, dim=-1)

    if return_n_tokens:
        return tokenizer.decode(generated_sequence.squeeze(), skip_special_tokens=True), generated_sequence.shape[-1]
    else:
        return tokenizer.decode(generated_sequence.squeeze(), skip_special_tokens=True)


def generate_M3ID(
    model, tokenizer, inputs,
    max_new_tokens=128, do_sample=False, raw_image=None, return_n_tokens=False,
    lamda=0.02, beta=0.1
):
    """
    Implements token-level generation with M3ID decoding.
    Uses two KV caches (with/without image) and applies mutual information adjustment.
    """
    model.eval()
    generated_tokens = []

    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    attention_mask = inputs.get("attention_mask", None)
    image_sizes = inputs.get("image_sizes", None)

    if image_sizes is None:
        assert raw_image is not None
        image_sizes = [(raw_image.height, raw_image.width)]

    past_key_values_c = None  # with image
    past_key_values_u = None  # without image
    next_token_id = None
    step = 1

    for _ in range(max_new_tokens):
        is_first_step = step == 1
        current_input_ids = input_ids if is_first_step else next_token_id

        # === Conditioned forward (with image) ===
        with torch.no_grad():
            outputs_c = model(
                input_ids=current_input_ids,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
                past_key_values=past_key_values_c,
                use_cache=True
            )
            logits_c = outputs_c.logits[:, -1, :]
            past_key_values_c = outputs_c.past_key_values
        
        attention_mask_u = attention_mask

        # === Unconditioned forward (no image) ===
        with torch.no_grad():
            outputs_u = model(
                input_ids=current_input_ids,
                pixel_values=None,
                image_sizes=None,
                attention_mask=attention_mask_u,
                past_key_values=past_key_values_u,
                use_cache=True
            )
            logits_u = outputs_u.logits[:, -1, :]
            past_key_values_u = outputs_u.past_key_values

        # === M3ID decoding adjustment ===
        gamma_t = math.exp(-lamda * step)
        lc = F.log_softmax(logits_c, dim=-1)
        lu = F.log_softmax(logits_u, dim=-1)
        adjusted_logits = lc + ((1 - gamma_t) / gamma_t) * (lc - lu)

        cutoff = torch.log(torch.tensor(beta).to(adjusted_logits.device)) + logits_c.max(dim=-1, keepdim=True).values
        adjusted_logits = adjusted_logits.masked_fill(logits_c < cutoff, -float("inf"))

        # === Token selection ===
        if do_sample:
            raise NotImplementedError("'do_sample=True' is not implemented.")
        else:
            next_token_id = torch.argmax(adjusted_logits, dim=-1, keepdim=True)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

        generated_tokens.append(next_token_id)

        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)
            ], dim=-1)

        step += 1

    if not generated_tokens:
        return ""

    generated_sequence = torch.cat(generated_tokens, dim=-1)

    if return_n_tokens:
        return tokenizer.decode(generated_sequence.squeeze(), skip_special_tokens=True), generated_sequence.shape[-1]
    else:
        return tokenizer.decode(generated_sequence.squeeze(), skip_special_tokens=True)


def generate_AvisC(
    model, tokenizer, inputs,
    max_new_tokens=128, do_sample=False, raw_image=None,
    avisc_alpha=2.5, avisc_beta=0.1, layer_gamma=0.8, lamb=1.0, return_n_tokens=False,
):
    """
    Implements token-level generation with AvisC decoding.
    Dynamically identifies blind image tokens at the first generation step
    and zeroes out non-blind image patches for contrastive decoding.
    Optimized for memory efficiency with Hugging Face models.
    """
    model.eval()
    generated_tokens = []

    # Extract inputs
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    attention_mask = inputs.get("attention_mask", None)
    image_sizes = inputs.get("image_sizes", None)

    # Default image sizes if not provided
    if image_sizes is None:
        assert raw_image is not None, "raw_image must be provided if image_sizes is None"
        image_sizes = [(raw_image.height, raw_image.width)]

    # Initialize variables
    device = input_ids.device
    past_key_values = None
    past_key_values_masked = None
    next_token_id = None
    step = 0
    pixel_values_masked = None

    # Image token handling
    image_token_index = model.config.image_token_index
    img_token_indices = (input_ids[0] == image_token_index).nonzero(as_tuple=True)[0]

    # Vision tower configuration for patch mapping
    vision_tower = model.vision_tower[0] if isinstance(model.vision_tower, (list, tuple)) else model.vision_tower
    patch_size = vision_tower.config.patch_size
    num_patches = (pixel_values.shape[2] // patch_size) * (pixel_values.shape[3] // patch_size)

    # Generation loop
    for _ in range(max_new_tokens):
        is_first_step = step == 0
        current_input_ids = input_ids if is_first_step else next_token_id

        # Clean forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=current_input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                image_sizes=image_sizes,
                past_key_values=past_key_values,
                output_attentions=is_first_step,
                use_cache=True
            )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            if is_first_step:
                attentions = outputs.attentions
                del outputs  # Free memory immediately

                # Compute per-layer image attention score
                layer_img_scores = []
                for layer_attn in attentions:
                    avg_attn = layer_attn.mean(dim=1)[:, -1, :]  # [1, seq_len]
                    img_score = avg_attn[:, img_token_indices].sum()
                    layer_img_scores.append(img_score)

                # Select top-K layers using gamma
                layer_img_scores = torch.stack(layer_img_scores)
                normalized = layer_img_scores / layer_img_scores.sum()
                sorted_vals, sorted_idx = torch.sort(normalized, descending=True)
                cumulative = torch.cumsum(sorted_vals, dim=0)
                top_k = (cumulative < layer_gamma).sum().item() + 1
                selected_layers = sorted_idx[:top_k]

                # Aggregate attention from top-K layers
                combined_attn = torch.stack([
                    attentions[i].mean(dim=1)[:, -1, img_token_indices] for i in selected_layers
                ], dim=0).mean(dim=0)  # [1, num_img_tokens]

                # Thresholding to find mask index (non-blind tokens)
                mean = combined_attn.mean()
                std = combined_attn.std()
                threshold = mean + lamb * std
                mask_idx = img_token_indices[(combined_attn < threshold).squeeze()]

                # Create masked pixel_values for contrastive pass
                pixel_values_masked = pixel_values.clone()
                if mask_idx.numel() > 0:
                    # Map sequence-level indices to patch indices
                    patch_indices = mask_idx - img_token_indices[0]  # Normalize to 0-based patch indices
                    valid_patch_indices = patch_indices[patch_indices < num_patches]

                    # Convert patch indices to spatial coordinates
                    h_patches = pixel_values.shape[2] // patch_size
                    w_patches = pixel_values.shape[3] // patch_size
                    patch_y = (valid_patch_indices // w_patches) * patch_size
                    patch_x = (valid_patch_indices % w_patches) * patch_size

                    # Zero out patches corresponding to non-blind tokens
                    for y, x in zip(patch_y, patch_x):
                        pixel_values_masked[:, :, y:y+patch_size, x:x+patch_size] = 0

                # Free memory before masked pass
                del attentions
                gc.collect()
                torch.cuda.empty_cache()

            else:
                del outputs  # Free memory in subsequent steps

        # Masked forward pass
        with torch.no_grad():
            outputs_masked = model(
                input_ids=current_input_ids,
                pixel_values=pixel_values_masked,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
                past_key_values=past_key_values_masked,
                use_cache=True
            )
            logits_masked = outputs_masked.logits[:, -1, :]
            past_key_values_masked = outputs_masked.past_key_values
            del outputs_masked  # Free memory

        # Contrastive decoding
        cutoff = torch.log(torch.tensor(avisc_beta).to(logits.device)) + logits.max(dim=-1, keepdim=True).values
        adjusted_logits = (1 + avisc_alpha) * logits - avisc_alpha * logits_masked
        adjusted_logits = adjusted_logits.masked_fill(logits < cutoff, -float("inf"))

        # Token selection
        if do_sample:
            raise NotImplementedError("'do_sample=True' is not implemented.")
        else:
            next_token_id = torch.argmax(adjusted_logits, dim=-1, keepdim=True)

        # Check for EOS token
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        generated_tokens.append(next_token_id)

        # Update attention mask
        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)
            ], dim=-1)

        step += 1

    # Return decoded sequence
    if not generated_tokens:
        return ""
    generated_sequence = torch.cat(generated_tokens, dim=-1)

    if return_n_tokens:
        return tokenizer.decode(generated_sequence.squeeze(), skip_special_tokens=True), generated_sequence.shape[-1]
    else:
        return tokenizer.decode(generated_sequence.squeeze(), skip_special_tokens=True)