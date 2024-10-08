import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2GroupNormConvLayer,Wav2Vec2LayerNormConvLayer,Wav2Vec2NoLayerNormConvLayer
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2FeatureEncoder

### Remark: this is a temporary file, to try to improve the behavior of Wav2vec2 to silence...
# right now it has no effects


class robustWav2Vec2LayerNormConvLayer(Wav2Vec2LayerNormConvLayer):
    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)

        ### Careful the following has not been verified as I use Wav2Vec2GroupNormConvLayer
        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class robustWav2Vec2GroupNormConvLayer(Wav2Vec2GroupNormConvLayer):
    def forward(self, hidden_states):

        hidden_states = self.conv(hidden_states)

        #Make sure that the hidden states are not artificially increased  by the presence of large silence block:
        # r1 = torch.sum(hidden_states==0,dim=-1,keepdim=True)
        # r2 = torch.sum(hidden_states!=0,dim=-1,keepdim=True)
        # ratio = (r2+r1)/(r2)
        # robust_mean = torch.mean(hidden_states,dim=-1,keepdim=True)*ratio
        # robust_var = torch.sqrt(torch.mean(torch.square(hidden_states-robust_mean),dim=-1,keepdim=True)*ratio+self.layer_norm.eps)
        #
        # to_correct = hidden_states[0,0,:] != 0 # TOFIX
        # hidden_states[:,:,to_correct] = (hidden_states[:,:,to_correct] - robust_mean)/(robust_var)
        # hidden_states[:,:,to_correct] = hidden_states[:,:,to_correct]*self.layer_norm.weight[None,:,None] + self.layer_norm.bias[None,:,None]

        ## normalization by block:
        # last_size = hidden_states.shape[-1]
        # cpad = torch.nn.ConstantPad1d((0,160-(last_size%160)),torch.mean(hidden_states[:,:,(last_size//160)*160:]))
        # hs = cpad(hidden_states)
        # hs = hs.reshape((hidden_states.shape[0],hidden_states.shape[1],-1,160)).transpose(2,1)
        # hs = hs.reshape(-1,hidden_states.shape[1],hs.shape[-1])
        # hs = self.layer_norm(hs).reshape((hidden_states.shape[0],-1,hidden_states.shape[1],hs.shape[-1]))
        # hs = hs.transpose(2,1).reshape(hidden_states.shape[0],hidden_states.shape[1],-1)
        # hs = hs[...,:-(160-(last_size%160))]
        #
        # ## Normalize over block of non-zero:
        # for i in range(hidden_states.shape[0]):
        #     is_non_zero = torch.where(torch.any(hidden_states[i:i+1,...]!=0,dim=1))
        #     block_end = torch.cat([is_non_zero[1][:-1][torch.diff(is_non_zero[1])>2],is_non_zero[1][-1:]])
        #     block_start = torch.cat([is_non_zero[1][0:1],is_non_zero[1][1:][torch.diff(is_non_zero[1])>2]])
        #     for bs,be in zip(block_start,block_end):
        #         hidden_states[i:i+1,:,bs:be+1] = self.layer_norm(hidden_states[i:i+1,:,bs:be+1])
        #     ## Apply layer norm over silences:
        #     hidden_states[i:i+1,:,torch.all(hidden_states[i,...]==0,dim=0)] = self.layer_norm(hidden_states[i:i+1,:,:])[:,:,torch.all(hidden_states[i,...]==0,dim=0)]

        # import matplotlib.pyplot as plt
        # fig,ax = plt.subplots()
        # ax.plot(torch.mean(hidden_states[0,0:1,:],dim=0).cpu())
        # # ax.set_xlim(9800,10000)
        # fig.show()
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

class robustWav2Vec2FeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""
    def __init__(self, config):
        super().__init__()

        if config.feat_extract_norm == "group":
            conv_layers = [robustWav2Vec2GroupNormConvLayer(config, layer_id=0)] + [
                Wav2Vec2NoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            conv_layers = [
                robustWav2Vec2LayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)
            ]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        self.conv_layers = nn.ModuleList(conv_layers)
        self.gradient_checkpointing = False
        self._requires_grad = True

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(self, input_values):
        hidden_states = input_values[:, None]

        # make sure hidden_states require grad for gradient_checkpointing
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        for conv_layer in self.conv_layers:
            if self._requires_grad and self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(conv_layer),
                    hidden_states,
                )
            else:
                hidden_states = conv_layer(hidden_states)

        return hidden_states
