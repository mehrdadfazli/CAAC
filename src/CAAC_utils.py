import torch
import torch.nn as nn
import numpy as np
from functools import partial
import logging
import math

logger = logging.getLogger(__name__)

def compute_attention_factor(exp_config, p=1):
    """
    Compute the dynamic attention scaling factor.
    """
    lamb = p * exp_config['min_lamb'] + (1 - p) * exp_config['max_lamb']
    
    return max(lamb, 1)

def get_calibration_vactor(data, beta=0.5):
    """
    Transforms a 2D signal towards uniformity and calculates the calibration vector.
    """
    if not 0 <= beta <= 1:
        raise ValueError("Beta must be between 0 and 1.")
    row_sums = torch.sum(data, dim=1, keepdim=True)
    row_averages = row_sums / data.shape[1]
    uniform_distribution = torch.ones_like(data) * row_averages
    transformed_data = (1 - beta) * data + beta * uniform_distribution
    calibration_vectors = transformed_data / data
    return calibration_vectors.mean(dim=0)

def attn_calib_compute(attn_maps, dim1_range, dim2_range, beta=0.7, avg_dim2=True):
    """
    Compute calibration matrices for attention maps.
    """
    calibration_matrices = {}
    with torch.no_grad():
        for layer_idx in range(len(attn_maps.keys())):
            attn_heads = attn_maps[f'language_model.model.layers.{layer_idx}.self_attn'][0]
            calibration_matrices[layer_idx] = []
            for head_idx, attn_map in enumerate(attn_heads):
                attn_img = attn_map[dim1_range][:, dim2_range]
                calib_vector = get_calibration_vactor(attn_img, beta)
                calibration_matrices[layer_idx].append(calib_vector)
    return calibration_matrices

def get_calibration_filter(attn_map, cal_vec, row_idx, col_idx):
    """
    Create a calibration filter for attention weights.
    """
    assert len(cal_vec) == len(col_idx), f"Column indices length ({len(col_idx)}) differs from calibration vector length ({len(cal_vec)})"
    cal_filter = torch.ones(attn_map.shape, dtype=torch.float16, device=attn_map.device)
    for r in row_idx:
        cal_filter[r][col_idx] = cal_vec
    return cal_filter

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Apply rotary position embeddings to query and key tensors.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key-value heads for multi-query attention.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    """
    Rotate half the hidden dimensions of the input.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class SelfAttentionModifier:
    """
    Modifies self-attention mechanisms with upscaling and calibration interventions.
    """
    def __init__(self, model, exp_config, img_token_idxs):
        self.model = model
        self.exp_config = exp_config
        self.img_token_idxs = img_token_idxs
        self.original_forwards = {}
        self.calibration_matrices = None
        self.dynamic_factor = None

    def update_calibration_matrix(self, attention_maps, input_image, processor, model_type):
        """
        Update calibration matrices based on attention maps.
        """
        from model_utils import process_inputs  # Local import to avoid circular dependency
        try:
            attention_maps.clear()
            img_token_id = self.model.config.image_token_index
            query = self.exp_config.get('calibration_query', "default query")
            inputs = process_inputs(input_image, query, processor, model_type)
            self.img_token_idxs = torch.nonzero(inputs['input_ids'][0] == img_token_id, as_tuple=False).flatten().cpu()
            outputs_ = self.model(**inputs, output_attentions=True)
            token_id = outputs_.logits.detach().cpu()[0, -1].argmax().item()
            logit = outputs_.logits[:, -1, token_id]
            logit.backward(retain_graph=True)
            
            if hasattr(outputs_, "language_model_outputs") and hasattr(outputs_.language_model_outputs, "attentions") and outputs_.language_model_outputs.attentions:
                attn_tuple = outputs_.language_model_outputs.attentions
            elif hasattr(outputs_, "attentions") and outputs_.attentions:
                attn_tuple = outputs_.attentions
            else:
                raise ValueError("Attention maps are empty.")
            
            for i, attn in enumerate(attn_tuple):
                attention_maps[f"language_model.model.layers.{i}.self_attn"] = attn
            
            row_range = torch.tensor(self.exp_config['input_token_idx_calibration'])
            column_range = self.img_token_idxs
            self.calibration_matrices = attn_calib_compute(attention_maps, row_range, column_range, self.exp_config['beta'])
            return self.calibration_matrices
        except Exception as e:
            logger.error(f"Error updating calibration matrix: {e}")
            raise

    def modify_attention_forward(self, self_attn, hidden_states, attention_mask=None, position_ids=None,
                               past_key_value=None, output_attentions=False, use_cache=False,
                               cache_position=None, position_embeddings=None, **kwargs):
        """
        Modified self-attention forward with upscaling and calibration interventions.
        """
        rotary_emb = self_attn.rotary_emb
        bsz, q_len, _ = hidden_states.size()
        query_states = self_attn.q_proj(hidden_states)
        key_states = self_attn.k_proj(hidden_states)
        value_states = self_attn.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, -1, self_attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self_attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self_attn.head_dim).transpose(1, 2)
        
        if position_embeddings is None:
            cos, sin = rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self_attn.layer_idx, cache_kwargs
            )
        
        key_states = repeat_kv(key_states, self_attn.num_key_value_groups)
        value_states = repeat_kv(value_states, self_attn.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self_attn.head_dim)
        
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        
        if self_attn.layer_idx in self.exp_config.get('img_txt_cal_layers', []):
            factor = self.dynamic_factor
            if factor is None:
                factor = self.exp_config['compute_attention_factor'](self.exp_config, 1)
            attn_weights[..., self.img_token_idxs] *= factor
        
        if self_attn.layer_idx in self.exp_config.get('img_cal_layers', []) and self.calibration_matrices is not None:
            col_idx = self.img_token_idxs.tolist()
            if attn_weights.shape[-2] == 1:
                row_idx = [0]
            else:
                row_idx = list(range(self.img_token_idxs[-1].item() + 1, attn_weights.shape[-2]))
            cal_vecs = self.calibration_matrices.get(self_attn.layer_idx, None)
            if cal_vecs is not None:
                cal_vecs_tensor = torch.stack(cal_vecs)
                cal_filter = torch.ones_like(attn_weights, dtype=torch.float16, device=attn_weights.device)
                cal_vecs_expanded = cal_vecs_tensor[None, :, None, :].expand(bsz, -1, len(row_idx), len(col_idx))
                row_idx_tensor = torch.tensor(row_idx, device=attn_weights.device)
                col_idx_tensor = torch.tensor(col_idx, device=attn_weights.device)
                row_idx_grid, col_idx_grid = torch.meshgrid(row_idx_tensor, col_idx_tensor, indexing='ij')
                cal_filter[:, :, row_idx_grid, col_idx_grid] = cal_vecs_expanded
                attn_weights = attn_weights * cal_filter
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self_attn.attention_dropout, training=self_attn.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self_attn.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value

    def register_hooks(self):
        """
        Register modified attention forward functions for specified layers.
        """
        layers_to_hook = set(self.exp_config.get('img_txt_cal_layers', [])) | set(self.exp_config.get('img_cal_layers', []))
        for l in layers_to_hook:
            layer = self.model.language_model.model.layers[l].self_attn
            self.original_forwards[l] = layer.forward
            layer.forward = partial(self.modify_attention_forward, layer)

    def remove_hooks(self):
        """
        Restore original self-attention forward functions.
        """
        for l, original_fn in self.original_forwards.items():
            self.model.language_model.model.layers[l].self_attn.forward = original_fn
        self.original_forwards.clear()