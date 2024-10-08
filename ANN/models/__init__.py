from typing import Dict,Type
from ANN.models.api import  forPostAnalysis
from ANN.models.wav2vec2.postAnalysis.forLoss import Wav2vec2_forLoss_ConstrainedMask

def _getpostmodels() -> Dict[str,Dict[str,Type[forPostAnalysis]]]:
    POST_MODELS = {"loss" : {"wav2vec2_ConstrainedMask": Wav2vec2_forLoss_ConstrainedMask}}
    return POST_MODELS
POST_MODELS = _getpostmodels()