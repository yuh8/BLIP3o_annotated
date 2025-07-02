from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from blip3o.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_IDX, UND_IMAGE_TOKEN_IDX

from .multimodal_encoder.builder import build_dit, build_gen_vision_tower
from .multimodal_projector.builder import build_down_projector


class blip3oMetaModel:
    def __init__(self, config):
        super(blip3oMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            # self.vision_tower = build_vision_tower(config, delay_load=True)
            # self.mm_projector = build_vision_projector(config)
            self.down_projector = build_down_projector(config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

        if hasattr(config, "gen_vision_tower"):
            self.gen_vision_tower = build_gen_vision_tower(config, delay_load=True)
            # self.gen_projector = build_gen_vision_projector(config)
            self.latent_queries = nn.Parameter(torch.randn(1, config.n_query, config.hidden_size))
            print(f" latent query size {self.latent_queries.shape}")

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

            self.dit, self.noise_scheduler = build_dit(config)

    # def get_vision_tower(self):
    #     vision_tower = getattr(self, 'vision_tower', None)
    #     if type(vision_tower) is list:
    #         vision_tower = vision_tower[0]
    #     return vision_tower

    def get_gen_vision_tower(self):
        gen_vision_tower = getattr(self, "gen_vision_tower", None)
        if type(gen_vision_tower) is list:
            gen_vision_tower = gen_vision_tower[0]
        return gen_vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        gen_vision_tower = model_args.gen_vision_tower

        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature

        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        pretrain_gen_mlp_adapter = model_args.pretrain_gen_mlp_adapter

        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.gen_vision_tower = gen_vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if getattr(self, "dit", None) is None:
            print("random initiation the DiT !!!")
            self.dit, self.noise_scheduler = build_dit(model_args)
        else:
            print("DiT load from checkpoint!!!")
            for p in self.dit.parameters():
                p.requires_grad = True

        if self.get_gen_vision_tower() is None:
            gen_vision_tower = build_gen_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.gen_vision_tower = [gen_vision_tower]
            else:
                self.gen_vision_tower = gen_vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                gen_vision_tower = self.gen_vision_tower[0]
            else:
                gen_vision_tower = self.gen_vision_tower
            gen_vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        # self.config.gen_projector_type = getattr(model_args, 'gen_projector_type', 'linear')

        self.config.gen_hidden_size = gen_vision_tower.hidden_size

        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type
        self.config.n_query = model_args.n_query
        self.config.gen_pooling = model_args.gen_pooling

        # if getattr(self, 'mm_projector', None) is None:
        #     print("random initiation the mm_project !!!")
        #     self.mm_projector = build_vision_projector(self.config)

        #     if 'unpad' in mm_patch_merge_type:
        #         embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
        #         self.image_newline = nn.Parameter(
        #             torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
        #         )
        # else:
        #     # In case it is frozen by LoRA
        #     for p in self.mm_projector.parameters():
        #         p.requires_grad = True

        if getattr(self, "down_projector", None) is None:
            print("random initiation the down_projector !!!")
            self.down_projector = build_down_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.down_projector.parameters():
                p.requires_grad = True

        if getattr(self, "latent_queries", None) is None:
            print("random initiation the latent_queries !!!")
            self.latent_queries = nn.Parameter(torch.randn(1, self.config.n_query, self.config.hidden_size))
        else:
            print("latent_queries load from checkpoint!!!")
            self.latent_queries.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            # self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class blip3oMetaForCausalLM(ABC):
    """
    This class uses the blip3oLlamaModel as self.model
    blip3oLlamaModel has implemented self.latent_queries = nn.Parameter(torch.randn(1, config.n_query, config.hidden_size))
    that is why we can get latent_querys
    """

    @abstractmethod
    def get_model(self):
        """
        later was implemented in blip3oLlamaForCausalLM
        """
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_gen_vision_tower(self):
        return self.get_model().get_gen_vision_tower()

    def encode_image(self, images):
        # breakpoint()
        gen_vision_tower = self.get_gen_vision_tower()
        device = gen_vision_tower.device
        images = images.to(device)
        prompt_image_embeds = gen_vision_tower(images)
        if "early" in self.get_gen_pooling():
            prompt_image_embeds = self.pool_img(prompt_image_embeds)
        num_img, _, c = prompt_image_embeds.shape
        # prompt_image_embeds = prompt_image_embeds.contiguous().view(-1, c)

        # ------------- compute similarity -------
        all_dist = 0
        count = 0
        for i in range(2, prompt_image_embeds.shape[1] - 1):
            diff = prompt_image_embeds[:, i, :].unsqueeze(1) - prompt_image_embeds[:, :i, :]
            dist = torch.sqrt(diff.square().sum(-1)).min().item()
            all_dist += dist
            count += 1
        all_dist /= count
        # self.dist = all_dist
        # print(self.dist)

        return prompt_image_embeds

    def get_mm_projector(self):
        return self.get_model().mm_projector

    def get_gen_projector(self):
        return None

    def get_n_query(self):
        return self.get_model().config.n_query

    def get_gen_pooling(self):
        return self.get_model().config.gen_pooling

    def pool_img(self, image_features):
        """
        Apply early 2D average pooling to spatial image features to reduce token resolution.

        This function is typically used in multimodal models to downsample the number of image tokens
        before feeding them into a decoder (e.g., diffusion model or language model). It assumes the
        input image features are arranged as flattened spatial tokens from a 2D vision backbone (e.g., ViT),
        and reshapes them into a 2D grid before applying average pooling.

        Pooling configuration is extracted from `self.get_gen_pooling()`, and expected to be in the form:
        `"early_pool2d_<stride>"` (e.g., "early_pool2d_4").

        Args:
            image_features (torch.Tensor): A tensor of shape [B, N, C], where:
                - B is the batch size,
                - N is the number of flattened spatial tokens (e.g., 576 from a 24x24 grid),
                - C is the channel dimension (e.g., 1024).

        Returns:
            torch.Tensor: A 4D tensor of shape [B, C, H_out, W_out] where:
                - H_out = W_out = original grid size // stride,
                - Typically used for further decoding or generation steps.
        """
        num_img, n, c = image_features.shape
        gen_pooling = self.get_gen_pooling()
        stride = int(gen_pooling.split("_")[-1])
        sqrt_n = int(n**0.5)
        image_features = image_features.permute(0, 2, 1).view(num_img, c, sqrt_n, sqrt_n)
        image_features = F.avg_pool2d(image_features, kernel_size=(stride, stride), stride=stride)
        return image_features

    def get_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        """
        Retrieves the noise scale (sigma) values for a given batch of diffusion timesteps.

        These sigmas represent the amount of noise injected at each diffusion step and are
        extracted from the model's noise scheduler. The returned tensor is reshaped to match
        the specified number of dimensions, enabling broadcasting for operations such as
        adding noise to latents during the forward diffusion process.

        Args:
            timesteps (torch.Tensor): A 1D tensor of integers representing diffusion timesteps for each sample in the batch.
            device (torch.device or str): The device to which the returned sigma tensor will be moved.
            n_dim (int, optional): The number of dimensions the output tensor should have.
                                Singleton dimensions are appended until this dimensionality is reached. Default is 4.
            dtype (torch.dtype, optional): The desired floating-point type of the returned tensor. Default is torch.float32.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, 1, ..., 1) with `n_dim` dimensions,
                        containing the noise scale (sigma) for each timestep in the batch.
        """
        sigmas = self.get_model().noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.get_model().noise_scheduler.timesteps.to(device=device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def mask_drop(self, latents, drop_prob=0.1):
        """
        Applies batch-level dropout to conditioning latents for classifier-free guidance.

        This method implements a simple form of stochastic conditioning removal by randomly zeroing
        out entire conditioning vectors (e.g., image or text embeddings) in a batch with probability `drop_prob`.
        This is used during training to enable the model to learn to generate outputs both with and
        without conditioning — a technique known as classifier-free guidance.

        For each item in the batch, a Bernoulli mask is sampled:
        - With probability `drop_prob`, the conditioning latent is replaced with zeros.
        - With probability `1 - drop_prob`, the latent is retained.

        This allows the model to later use both unconditional and conditional branches during generation,
        and interpolate between them at inference time (e.g., via guidance scale).

        Args:
            latents (torch.Tensor): Conditioning latent tensor of shape (B, ...) — typically
                                    output from a projection network (e.g., image encoder or text embedder).
            drop_prob (float): Probability of zeroing out each sample's conditioning vector in the batch.
                            Must be in [0, 1]. If <= 0, no dropout is applied.

        Returns:
            torch.Tensor: The latents tensor with some conditioning vectors zeroed out, shape unchanged.
        """
        if drop_prob <= 0:
            return latents
        mask = torch.bernoulli(torch.zeros(latents.shape[0], device=latents.device, dtype=latents.dtype) + drop_prob)
        while len(mask.shape) < len(latents.shape):
            mask = mask.unsqueeze(-1)
        mask = 1 - mask  # need to flip 0 <-> 1
        return latents * mask

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        gen_images,
        und_images,
        grid_thw,
        i_s_pos,
        image_sizes=None,
    ):
        vision_tower = self.visual
        gen_vision_tower = self.get_gen_vision_tower()
        if (gen_images is None and und_images is None) or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None, None

        prompt_image_embeds = gen_vision_tower(gen_images)  # TODO: check dimension

        # If 'early' pooling is specified (e.g., 'early_pool2d_4'), apply 2D average pooling early in the pipeline
        # to reduce spatial resolution of image features and save memory/computation downstream.
        # For example, input shape [B, 576, C] (i.e., 24×24 spatial grid) is reshaped to [B, C, 24, 24], then pooled to [B, C, 6, 6]
        # using stride=4, reducing the token count before they are consumed by the decoder/generator.
        if "early" in self.get_gen_pooling():
            prompt_image_embeds = self.pool_img(prompt_image_embeds)
        target_image_embeds = torch.clone(prompt_image_embeds).detach()
        latent_queries = self.get_model().latent_queries.repeat(input_ids.shape[0], 1, 1)
        H = latent_queries.shape[-1]
        latent_queries = latent_queries.contiguous().view(-1, H)

        if und_images is not None:
            und_image_embeds = vision_tower(und_images, grid_thw=grid_thw)

        image_idx = input_ids == IMAGE_TOKEN_IDX
        und_image_idx = input_ids == UND_IMAGE_TOKEN_IDX
        output_indicator = labels != -100
        input_indicator = labels == -100
        text_embeds = self.get_model().embed_tokens(input_ids)
        gen_img_idx = torch.logical_and(output_indicator, image_idx)

        text_embeds = text_embeds.clone()
        text_embeds[gen_img_idx] = latent_queries

        und_img_idx = torch.logical_and(input_indicator, und_image_idx)

        if und_images is not None:
            text_embeds[und_img_idx] = und_image_embeds.to(text_embeds.device)[: und_img_idx.sum(), :]

        labels[image_idx] = -100

        return None, position_ids, attention_mask, past_key_values, text_embeds, labels, target_image_embeds

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
