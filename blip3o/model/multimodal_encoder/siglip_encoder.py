import os
from dataclasses import dataclass
from functools import partial, reduce
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from PIL import Image
from torch import nn
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput


class SigLipImageProcessor:
    """
    Image processor for preparing image inputs for the SigLIP vision encoder.

    This processor handles conversion of input images (PIL or NumPy) into normalized,
    batched tensors suitable for deep learning models. It performs resizing, rescaling,
    normalization, and channel formatting, and returns results in a HuggingFace-compatible
    `BatchFeature` object.

    Args:
        image_mean (tuple): Mean values for each image channel used during normalization.
                            Defaults to (0.5, 0.5, 0.5).
        image_std (tuple): Standard deviation values for each channel used during normalization.
                           Defaults to (0.5, 0.5, 0.5).
        size (tuple): Target size (height, width) for resizing input images. Defaults to (384, 384).
        crop_size (dict, optional): Dictionary specifying crop height and width. Not used directly in this
                                    processor, but retained for API compatibility.
        resample (int): Resampling filter to use during resizing (e.g., PILImageResampling.BICUBIC).
                        Defaults to BICUBIC.
        rescale_factor (float): Scaling factor to convert pixel values (e.g., 1/255 for 0–255 to 0–1).
                                Defaults to 1/255.
        data_format (ChannelDimension): Channel format of the output tensor (e.g., ChannelDimension.FIRST
                                        for (C, H, W)). Defaults to FIRST.

    Methods:
        preprocess(images, return_tensors):
            Preprocesses a single image or a list of images and returns a `BatchFeature` containing
            the processed pixel values. Handles RGB conversion, resizing, rescaling, normalization,
            and grayscale fallback if needed.

            Args:
                images (PIL.Image.Image or list): Input image or list of images to preprocess.
                return_tensors (str): Type of tensor to return. One of "pt", "tf", "np", or "jax".

            Returns:
                BatchFeature: A HuggingFace-compatible dictionary-like object with key `"pixel_values"`
                              containing the processed batched tensor or list.
    """

    def __init__(
        self,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        size=(384, 384),
        crop_size: Dict[str, int] = None,
        resample=PILImageResampling.BICUBIC,  # interpolation method for resizing image
        rescale_factor=1 / 255,  # rescale image between [0, 1]
        data_format=ChannelDimension.FIRST,
    ):
        crop_size = crop_size if crop_size is not None else {"height": 384, "width": 384}
        # get_size_dict(384, default_to_square=True) ➜ dictionary {"height": 384, "width": 384}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    def preprocess(self, images, return_tensors):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            # to adapt video data
            images = [to_numpy_array(image) for image in images]
            assert isinstance(images, list)

        try:
            transforms = [
                convert_to_rgb,
                to_numpy_array,
                partial(
                    resize,
                    size=self.size,
                    resample=self.resample,
                    data_format=self.data_format,
                ),
                partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
                partial(
                    normalize,
                    mean=self.image_mean,
                    std=self.image_std,
                    data_format=self.data_format,
                ),
                partial(
                    to_channel_dimension_format,
                    channel_dim=self.data_format,
                    input_channel_dim=self.data_format,
                ),
            ]

            # "Reduce" to cumulative apply each transformation to every img in the list
            # [img1, img2, img3]
            #     ↓ apply convert_to_rgb
            # [img1_rgb, img2_rgb, img3_rgb]
            #     ↓ apply resize
            # [img1_resized, img2_resized, img3_resized]
            #     ↓ apply normalize
            # [img1_normalized, img2_normalized, img3_normalized]
            images = reduce(lambda x, f: [*map(f, x)], transforms, images)
            data = {"pixel_values": images}
        except ValueError:
            # processing gray-scaled images
            try:
                transforms = [
                    convert_to_rgb,
                    to_numpy_array,
                    partial(
                        resize,
                        size=self.size,
                        resample=self.resample,
                        data_format=self.data_format,
                    ),
                    partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
                    partial(
                        normalize,
                        mean=self.image_mean[0],
                        std=self.image_std[0],
                        data_format=self.data_format,
                    ),
                    partial(
                        to_channel_dimension_format,
                        channel_dim=self.data_format,
                        input_channel_dim=self.data_format,
                    ),
                ]
                images = reduce(lambda x, f: [*map(f, x)], transforms, images)
                processed_images = [np.repeat(img, repeats=3, axis=0) for img in images]
                data = {"pixel_values": processed_images}
            except ValueError as e:
                print(f"Grayscale processing failed: {e}")

        # Stack all images in the list to a pt tenstor and assign it as value to "pixel_values" key
        return BatchFeature(data=data, tensor_type=return_tensors)


class SigLipVisionConfig(PretrainedConfig):
    """
    Configuration class for the SigLip vision model.

    Stores model architecture parameters and preprocessing configurations.

    Args:
        hidden_size (int): Dimensionality of the encoder layers and the pooler layer.
        image_mean (tuple): Mean values for each image channel used in normalization.
        intermediate_size (int): Dimensionality of the "intermediate" (i.e., feed-forward) layer.
        num_hidden_layers (int): Number of hidden layers in the Transformer encoder.
        num_attention_heads (int): Number of attention heads for each attention layer.
        num_channels (int): Number of input channels in the images.
        image_size (int): Height and width of the input images.
        patch_size (int): Size of the patches to divide the image into.
        hidden_act (str): Activation function used in the encoder and pooler.
        layer_norm_eps (float): Epsilon value for layer normalization.
        attention_dropout (float): Dropout rate for attention probabilities.
        **kwargs: Additional keyword arguments.
    """

    model_type = "siglip_vision_model"

    def __init__(
        self,
        hidden_size=1152,
        image_mean=(0.5, 0.5, 0.5),
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        image_size=384,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.image_mean = image_mean

    @classmethod  # classmethod as an alternative constructor
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        """
        Load a configuration from a pretrained model or directory.

        This class method serves as an "alternative constructor" for instantiating
        configuration classes (like `SigLipVisionConfig`) using a pretrained model
        identifier or local configuration file. It handles the logic for locating,
        downloading (if necessary), and parsing the configuration, returning an
        instance of the class populated with the appropriate parameters.

        Args:
            pretrained_model_name_or_path (Union[str, os.PathLike]):
                The identifier of a pretrained model (e.g., on Hugging Face Hub)
                or the path to a directory containing a `config.json` file.
            **kwargs:
                Additional keyword arguments to pass to the configuration constructor
                or to override specific values in the loaded config.

        Returns:
            An instance of `cls` (typically a subclass of `PretrainedConfig`),
            initialized with the parameters from the configuration file.

        Why classmethod:
            This method is marked as a `@classmethod` because it needs to instantiate
            the class (`cls`) without requiring an existing instance. This allows it
            to act as an alternative constructor and ensures that subclass-specific
            behavior is preserved when invoked from derived classes. It encapsulates
            complex loading logic and supports extensibility, inheritance, and
            modular configuration management in a clean and reusable way.
        """
        # for assigning HF hub authentication token
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from SigLipConfig
        if config_dict.get("model_type") == "siglip":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            print(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPVisionModelOutput with CLIP->SigLip
class SigLipVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.
    loss = outputs.loss
    logits = outputs.logits

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SigLipVisionEmbeddings(nn.Module):
    """
    Converts an input image into a sequence of patch embeddings with positional context
    for the Vision Transformer component of the SigLIP model.
    It uses a convolution operation with both kernel size and stride size equal to patch_size

    Args:
        config (SigLipVisionConfig):
            Configuration object containing:
            - hidden_size: Output dimensionality of each patch embedding.
            - image_size: Height and width of the input images.
            - patch_size: Dimension of each patch (assumed square).
            - num_channels: Number of input image channels (e.g., 3 for RGB).

    Attributes:
        patch_embedding (nn.Conv2d): A convolution layer with kernel size and stride
            equal to patch_size, projecting non-overlapping image patches into embedding vectors.
        position_embedding (nn.Embedding): Learnable embeddings representing the position
            of each patch in the image grid.
    """

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        # Register non-trainable, device-aware position IDs buffer.
        # Marked non-persistent since it can be regenerated at runtime
        # (e.g., for dynamic input shapes or varying patch counts).
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # [B, C, H, W] --> [B, embed_dim, H/patch_size, W/patch_size]
        patch_embeds = self.patch_embedding(pixel_values)
        # [B, embed_dim, H/patch_size, W/patch_size] --> [B, embed_dim, H/patch_size * W/patch_size]
        # [B, H/patch_size * W/patch_size, embed_dim]
        embeddings = patch_embeds.flatten(2).transpose(1, 2).contiguous()

        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SigLipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention.__init__
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Prefer reshape over view — it handles non-contiguous tensors safely
        # by copying if needed, while view requires contiguous memory and may error.
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        k_v_seq_len = key_states.shape[-2]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # Upcast to float32 for numerically stable softmax, as bfloat16/float16
        # lack sufficient mantissa precision to safely compute exponentials;
        # then downcast back to match model dtype.
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # data matrix are laid out in row major fashion, transpose ops breaks this
        # that is why we need contiguous here
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->SigLip
class SigLipMLP(nn.Module):
    """
    Feed-forward neural network ("MLP block") used within the SigLip model,
    typically following a self-attention or cross-attention layer.

    Consists of two linear transformations separated by a configurable non-linear activation.

    Args:
        config: Configuration object (SigLipVisionConfig or equivalent) containing:
            - hidden_size (int): size of the model's hidden representation.
            - intermediate_size (int): size of the hidden layer in the MLP.
            - hidden_act (str): name of the activation function (e.g., 'gelu', 'relu').

    Layers:
        - fc1 (nn.Linear): projects from `hidden_size` to `intermediate_size`.
        - activation_fn (callable): activation function selected via `ACT2FN[config.hidden_act]`.
        - fc2 (nn.Linear): projects back from `intermediate_size` to `hidden_size`.

    Forward pass:
        - Applies `fc1`, then activation, then `fc2`.
        - Transforms input of shape `(batch, seq_len, hidden_size)` and outputs the same shape.

    Returns:
        Tensor of shape `(batch, seq_len, hidden_size)`, suitable for residual connections.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer with CLIP->SigLip
class SigLipEncoderLayer(nn.Module):
    """
    A single Transformer encoder layer used in the SigLIP vision model.
    This takes places after the conv2d based patchifying of the input image


    This layer applies:
      1. A layer normalization (`layer_norm1`)
      2. A self-attention block (`SigLipAttention`) with optional attention outputs
      3. A residual connection
      4. Another layer normalization (`layer_norm2`)
      5. A two-layer MLP block (`SigLipMLP`)
      6. A second residual connection

    Args:
        config (SigLipVisionConfig):
            Configuration object containing:
              - hidden_size: dimensionality of hidden representations
              - layer_norm_eps: epsilon value for layer normalization
    """

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SigLipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        # this is to output attention weights as well
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class SigLipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.

    SigLipPreTrainedModel is a template class that connects your SigLip model to the Hugging Face machinery:
        Declares which config it's compatible with
        Enables gradient checkpointing
        Defines a hook (_init_weights) where you initialize components
        Inherits essential loading/saving capabilities for easy pretrained model usage
    """

    config_class = SigLipVisionConfig
    base_model_prefix = "siglip"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        pass


# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with CLIP->SigLip
class SigLipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SigLipEncoderLayer`].

    Args:
        config: SigLipVisionConfig
    """

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SigLipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        # return a standar dataclass
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class SigLipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)

    def forward(self, hidden_state):
        # hidden_state: [batch_size, seq_len, hidden_size]
        batch_size = hidden_state.shape[0]
        # Create probe query: repeat the learnable probe for each batch
        # self.probe is [1, 1, hidden_size] → probe is [batch_size, 1, hidden_size]
        probe = self.probe.repeat(batch_size, 1, 1)  # → [B, 1, H]

        # Apply multi-head attention:
        # Query   = probe:      [B, 1, H]
        # Key     = hidden_state: [B, seq_len, H]
        # Value   = hidden_state: [B, seq_len, H]
        # output: attention returns a tuple (attn_output, attn_weights)
        # attn_output: [B, 1, H]
        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]  # → [B, 1, H]

        # Save residual for skip connection
        residual = hidden_state  # [B, 1, H]

        # Normalize
        hidden_state = self.layernorm(hidden_state)  # [B, 1, H]

        # Apply MLP block:
        # - Input: [B, 1, H]
        # - Through MLP: project to intermediate_size then back → stays [B, 1, H]
        hidden_state = residual + self.mlp(hidden_state)  # → [B, 1, H]

        # Return the pooled vector, removing the sequence dimension
        # hidden_state[:, 0]: [B, H]
        return hidden_state[:, 0]


class SigLipVisionTransformer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        # patchify images and then encod each into a vision embedding vector using conv2d ops
        self.embeddings = SigLipVisionEmbeddings(config)
        # transformer encoder layers after patchifying
        self.encoder = SigLipEncoder(config)
        # layer norm transformer outputs
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        # attention based pooling with a learnable query vector
        self.head = SigLipMultiheadAttentionPoolingHead(config)

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = self.head(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SigLipVisionModel(SigLipPreTrainedModel):
    config_class = SigLipVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["SigLipEncoderLayer"]

    def __init__(self, config: SigLipVisionConfig):
        super().__init__(config)

        self.vision_model = SigLipVisionTransformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, SigLipVisionModel

        >>> model = SigLipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled features
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class SigLipVisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False):
        super().__init__()

        self.is_loaded = False

        # instantiating config to be passed in for other module
        # we can also use an alternative constructor .from_pretrained to do the instantiation
        self.config = SigLipVisionConfig()

        self.vision_tower_name = vision_tower

        self.image_processor = SigLipImageProcessor()

        if not delay_load:
            # rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            # rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(vision_tower_cfg, "mm_tunable_parts") and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts:
            # rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            # for frozen model, we only load config
            self.cfg_only = self.config

    def load_model(self, device_map=None):
        if self.is_loaded:
            # rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        self.vision_tower = SigLipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)

        # del self.vision_tower.vision_model.encoder.layers[-1:]
        # removed attention pooling for connecting vision tokens to diffuser
        self.vision_tower.vision_model.head = nn.Identity()
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                # image_feature = image_forward_out.hidden_states[-1].to(image.dtype)
                image_feature = image_forward_out.last_hidden_state.to(image.dtype)
                assert image_features.shape[-2] == 729
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            # image_features = image_forward_outs.hidden_states[-1].to(images.dtype)
            image_features = image_forward_outs.last_hidden_state.to(images.dtype)
            assert image_features.shape[-2] == 729

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size
        # return self.model_config["vision_cfg"]["image_size"] // self.model_config["vision_cfg"]["patch_size"]

    @property
    def image_size(self):
        return self.config.image_size
