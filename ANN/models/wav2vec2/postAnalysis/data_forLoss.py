import numpy as np
from typing import  Dict, List, Union
import torch
import torch.utils.checkpoint

from dataclasses import dataclass
from typing import Optional
from transformers import Wav2Vec2FeatureExtractor, BatchFeature
from torch.utils.data import Dataset

@dataclass
class DataCollatorForWav2Vec2Pretraining_withPreprocesing_fixedMask:
    """
    A DataCollator that can be used with soundmat type dataset.
    It preprocesses by normalizing the data, # truncating too long sequences and padding.
    ## Pierre 18/07/2023: we remove the truncation

    ## Pierre 01/08/2023: we want to perform cached-preprocessing of the inputs
    # In order to do that, we feed None to the preprocessing if needs be.
    """
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    remove_normalization : bool = False ## forces no-sound normalization by the feature extractor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # No safe-checking on the sampling rate, make sure it is 16000 before...
        def iter_not_None(x):
            for a in x:
                if a is not None:
                    yield  a
        def torch_to_numpy(x):
            if isinstance(x,torch.Tensor):
                return x.numpy()
            return x
        if self.remove_normalization:
            self.feature_extractor.do_normalize = False
        batch = self.feature_extractor(
            raw_speech = [torch_to_numpy(f["input_values"]) for f in iter_not_None(features)],
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            sampling_rate=16000,
            # Truncation parameters for the sounds!
            truncation= False,
            # max_length = int(20* self.feature_extractor.sampling_rate),
            # min_length = int(2*self.feature_extractor.sampling_rate)

            # return_attention_mask=True,
            # sampling_rate = 16000
        )
        ## Important remark:
        # we don't use attention mask to measure where the padding occurs,
        # because we use pretrained models that did not use it during their training.
        # (see huggingface doc: https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor)

        # assert isinstance(features[0]["mask_time_indices"], np.ndarray)

        mti = self.feature_extractor.pad(
            processed_features=BatchFeature({"input_values": [torch_to_numpy(f["mask_time_indices"]) for f in iter_not_None(features)]}),
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=False,
            return_tensors="pt"
        )
        batch["mask_time_indices"] = mti["input_values"]

        sni = self.feature_extractor.pad(
            processed_features=BatchFeature({"input_values": [torch_to_numpy(f["sampled_negative_indices"]) for f in iter_not_None(features)]}),
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=False,
            return_tensors="pt"
        )
        ## Correct for the batch size:
        batch["sampled_negative_indices"] = sni["input_values"].to(torch.long)

        if 'latent_attention_mask' in features[0].keys():
            batch_features = BatchFeature(
                {"input_values": [torch_to_numpy(f["latent_attention_mask"]) for f in iter_not_None(features)]})
            lam = self.feature_extractor.pad(
                processed_features=batch_features,
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_attention_mask=False,
                return_tensors="pt",
            )
            batch["latent_attention_mask"] = lam["input_values"].to(torch.long)


        if 'latent_time_reduction' in features[0].keys():
            ltr = self.feature_extractor.pad(
                processed_features=BatchFeature({"input_values": [torch_to_numpy(f["latent_time_reduction"]) for f in iter_not_None(features)]}),
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_attention_mask=False,
                return_tensors="pt"
            )
            batch["latent_time_reduction"] = ltr["input_values"].to(torch.bool)

        return batch
def get_collator_withPreprocessing_fixedMask(file_configPreprocessor,remove_normalization: bool = False):
    feature_extractor = Wav2Vec2FeatureExtractor.from_json_file(file_configPreprocessor)
    data_collator = DataCollatorForWav2Vec2Pretraining_withPreprocesing_fixedMask(feature_extractor,
                                                                        remove_normalization=remove_normalization)
    return  data_collator
