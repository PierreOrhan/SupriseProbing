from ANN.models.api import forPostAnalysis
# from ANN.models.wav2vec2.utils_data import get_collator_withPreprocessing
from transformers import Wav2Vec2FeatureExtractor
from typing import Union,List,Dict,Tuple
from pathlib import  Path
import torch


def get_input_lengths(input_lengths, conv_kernel, conv_stride):
    # This function computes the temporal length of Wav2vec2's latent vector after downsampling by
    # the feature extractor, i.e the set of convolutional layers.
    # It takes as input conv_kernel and conv_stride, a list of the respective kernel siwe and stride of each
    # convolutional layers. These can be obtained from the .conf file along with the huggingface model.
    def torch_int_div(tensor1, tensor2):
        return torch.div(tensor1, tensor2, rounding_mode="floor")

    def _conv_out_length(input_length, kernel_size, stride):
        # 1D convolutional layer output length formula taken
        # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        return torch_int_div(input_length - kernel_size, stride) + 1

    for kernel_size, stride in zip(conv_kernel, conv_stride):
        input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
    return input_lengths

class Wav2vec2Post(forPostAnalysis):
    @classmethod
    def layerzip(cls) -> List[Tuple[str,Dict]]:
        return [("conv",{"nb_layer":7,"layer_size":512}),
                ("trans",{"nb_layer":13,"layer_size":768})]

    @classmethod
    def preprocessor_from_pretrained(cls,path : Union[str,Path]):
        return Wav2Vec2FeatureExtractor.from_pretrained(path)

    @classmethod
    def get_downsampleSize(self,input_size : int):
        # This function returns the number of element downsampled by the model in each of the layer
        # as a function of the size of the input.
        wav2vec2_params = {"conv_kernel": [10, 3, 3, 3, 3, 2, 2],
                           "conv_stride": [5, 2, 2, 2, 2, 2, 2]}
        latent_length = get_input_lengths(input_size, wav2vec2_params["conv_kernel"],
                                          wav2vec2_params["conv_stride"])
        return latent_length
