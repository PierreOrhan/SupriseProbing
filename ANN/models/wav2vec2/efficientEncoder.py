

## We observed that there is one
# very unefficient operation in the encoder as defined by huggingface,
# Where the attention_mask is repeated to set the inputs to 0
# This operation is not useful as in that case the padded inputs are set to 0
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2EncoderLayer,Wav2Vec2PositionalConvEmbedding,Wav2Vec2EncoderLayerStableLayerNorm,is_deepspeed_zero3_enabled,BaseModelOutput
from typing import Optional
import torch.nn as nn
import torch
import numpy as np


class fastWav2Vec2EncoderStableLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [Wav2Vec2EncoderLayerStableLayerNorm(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            ## Originally this function was to:
            # make sure padded tokens are not attended to
            # expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            # hidden_states[~expand_attention_mask] = 0
            ## --> We remove that operation to increase in efficiency:

            # Pierre, 17/07/2023
            # The problem of removing this setting to 0 is that the future hidden_states values
            # can be used by the relative convolutional encoder!!!
            # the next lines perform that operation more efficiently:
            # Finally we also deal the case where the attention_mask are provided as a 2D matrix and
            # should not be expanded

            # init_size = attention_mask.shape[-1]
            # attention_mask = attention_mask.to(dtype=torch.bool)
            # hidden_states = hidden_states[:,attention_mask[-1,:],:]
            # am = attention_mask[:,attention_mask[-1,:]]
            #
            # # hidden_states = attention_mask[..., None] * hidden_states
            # am = 1.0 - am[:, None, None, :].to(dtype=hidden_states.dtype)
            # am = am * torch.finfo(hidden_states.dtype).min
            # am = am.expand(
            #     am.shape[0], 1, am.shape[-1], am.shape[-1]
            # )
            if len(attention_mask.shape)==3:
                ## the latent attention is provided as a matrix
                attention_mask = 1.0 - attention_mask[:,None,:,:].to(dtype=hidden_states.dtype)
                attention_mask  = attention_mask * torch.finfo(hidden_states.dtype).min
            else:
                hidden_states = attention_mask[..., None] * hidden_states
                attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
                attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
                attention_mask = attention_mask.expand(
                    attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
                )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)

        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        for layer in self.layers:
            if output_hidden_states:
                # hs = torch.zeros(hidden_states.shape[0],init_size,hidden_states.shape[-1]).to(hidden_states)
                # hs[:,attention_mask[-1,:]] = hidden_states
                # all_hidden_states = all_hidden_states + (hs,)

                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                # XXX: could optimize this like synced_gpus in generate_utils but not sure if it's worth the code complication
                if self.gradient_checkpointing and self.training:
                    # create gradient checkpointing function
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        attention_mask,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        # hs = torch.zeros(hidden_states.shape[0], init_size, hidden_states.shape[-1]).to(hidden_states)
        # hs[:, attention_mask[-1, :]] = hidden_states
        # hidden_states = hs

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )



class fastWav2Vec2Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([Wav2Vec2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            ## Originally this function was to:
            # make sure padded tokens are not attended to
            # expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            # hidden_states[~expand_attention_mask] = 0
            ## --> We remove that operation to increase in efficiency:

            # Pierre, 17/07/2023
            # The problem of removing this setting to 0 is that the future hidden_states values
            # can be used by the relative convolutional encoder!!!
            # the next lines perform that operation more efficiently:
            # Finally we also deal the case where the attention_mask are provided as a 2D matrix and
            # should not be expanded

            hidden_states = attention_mask[..., None] * hidden_states
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    # create gradient checkpointing function
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        attention_mask,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
