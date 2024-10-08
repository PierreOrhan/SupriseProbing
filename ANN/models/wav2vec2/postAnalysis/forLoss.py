from ANN.models.wav2vec2.with_grad_mult import ModelOutput
from ANN.models.wav2vec2.postAnalysis.data_forLoss import  get_collator_withPreprocessing_fixedMask,\
    DataCollatorForWav2Vec2Pretraining_withPreprocesing_fixedMask
from ANN.models.api import forLossAnalysis
from ANN.models.wav2vec2.with_grad_mult import Wav2Vec2ForPreTraining_latentMask

from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2FeatureExtractor

from typing import Optional,Tuple,Dict,List,Union
from pathlib import Path
import torch.nn as nn
import torch
import torch.utils.checkpoint
from dataclasses import dataclass
from datasets import Dataset


@dataclass
class Wav2Vec2ForLossOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    projected_states: torch.FloatTensor = None
    projected_quantized_states: torch.FloatTensor = None
    codevector_perplexity: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    contrastive_loss: Optional[torch.FloatTensor] = None
    diversity_loss: Optional[torch.FloatTensor] = None
    pen_loss : Optional[torch.FloatTensor] = None
    pen_loss_nosum: Optional[torch.FloatTensor] = None
    contrastive_loss_notReduce : Optional[torch.FloatTensor] = None

class Wav2vec2_forLoss_ConstrainedMask(Wav2Vec2ForPreTraining_latentMask, forLossAnalysis):

    @classmethod
    def get_mainloss_name(cls):
        return "contrastive_loss_notReduce"

    @classmethod
    def preprocessor_from_pretrained(cls,path : Union[str,Path]):
        return Wav2Vec2FeatureExtractor.from_pretrained(path)

    @classmethod
    def layerzip(cls) -> List[Tuple[str,Dict]]:
        return [("conv",{"nb_layer":7,"layer_size":512}),
                ("trans",{"nb_layer":13,"layer_size":768})]

    @classmethod
    def get_collator(cls,file_configPreprocessor,remove_normalization: bool = False) -> DataCollatorForWav2Vec2Pretraining_withPreprocesing_fixedMask:
        return get_collator_withPreprocessing_fixedMask(file_configPreprocessor,remove_normalization)

    def get_downsampleSize(self,input_size : int):
        # This function returns the number of element downsampled by the model in each of the layer
        # as a function of the size of the input.
        raise Exception("not implemented")


    def preprocess_dataset(self,ds : Dataset) -> Dataset:
        return ds


    def forward(self,input_values,
                mask_time_indices : torch.BoolTensor,
                sampled_negative_indices: torch.LongTensor,
                latent_attention_mask: Optional[torch.BoolTensor] = None,
                latent_time_reduction : Optional[torch.BoolTensor] = None):

            # latent_attention_mask: a mask that is used in the transformer to
            # prevents some inputs of being used in the transformer computations

            # latent_time_reduction: similar to mask_time_indices but indicates which
            # tokens to use in the loss reduction. This could be useful if the masks length is above one token
            # and the final or initial token contain overlaps between two acoustic elements.

            assert not self.training

            with torch.no_grad():

                if mask_time_indices is not None:
                    mask_time_indices = mask_time_indices.to(torch.bool)

                outputs = self.wav2vec2(
                    input_values,
                    latent_attention_mask=latent_attention_mask,
                    output_attentions=None,
                    output_hidden_states=None,
                    mask_time_indices=mask_time_indices,
                    return_dict=None,
                )

                # 1. project all transformed features (including masked) to final vq dim
                transformer_features = self.project_hid(outputs[0])

                # 2. quantize all (unmasked) extracted features and project to final vq dim
                extract_features = self.dropout_features(outputs[1])
                features_pen = outputs[2]
                features_pen = features_pen * mask_time_indices.sum()

                quantized_features, codevector_perplexity = self.quantizer(
                    extract_features, mask_time_indices=mask_time_indices
                )
                quantized_features = self.project_q(quantized_features)

                loss = contrastive_loss = diversity_loss = None
                if sampled_negative_indices is not None:
                    batch_size, sequence_length, hidden_size = quantized_features.shape

                    # for training, we sample negatives
                    # 3. sample K negatives (distractors) quantized states for contrastive loss
                    # if attention_mask is passed, make sure that padded feature vectors cannot be sampled
                    # sample negative quantized vectors BTC => (BxT)C

                    ## Pierre 18/07/2023: we need to correct the sampled_negative_indices because we assume they are given without knowledge
                    # of the batch size:
                    for batch_idx in range(quantized_features.shape[0]):
                        sampled_negative_indices[batch_idx] += batch_idx * sequence_length


                    negative_quantized_features = quantized_features.view(-1, hidden_size)[
                        sampled_negative_indices.long().view(-1)
                    ]
                    ### Pierre: problem here because the negative indices are given without knowledge of the batch size
                    # so they can't be right here !!!!!!!!!

                    negative_quantized_features = negative_quantized_features.view(
                        batch_size, sequence_length, -1, hidden_size
                    ).permute(2, 0, 1, 3)

                    # 4. compute logits, corresponding to `logs = sim(c_t, [q_t, \sim{q}_t]) / \kappa`
                    # of equation (3) in https://arxiv.org/pdf/2006.11477.pdf
                    logits = self.compute_contrastive_logits(
                        quantized_features[None, :],
                        negative_quantized_features,
                        transformer_features,
                        self.config.contrastive_logits_temperature,
                    )

                    # 5. if a negative vector is identical to the positive (i.e. when codebook utilization is low),
                    # its cosine similarity will be masked
                    neg_is_pos = (quantized_features == negative_quantized_features).all(-1)

                    if neg_is_pos.any():
                        logits[1:][neg_is_pos] = float("-inf")

                    # 6. compute contrastive loss \mathbf{L}_m = cross_entropy(logs) =
                    # -log(exp(sim(c_t, q_t)/\kappa) / \sum_{\sim{q}} exp(sim(c_t, \sim{q})/\kappa))
                    logits = logits.transpose(0, 2).reshape(-1, logits.size(0))
                    target = ((1 - mask_time_indices.long()) * -100).transpose(0, 1).flatten()

                    contrastive_loss = nn.functional.cross_entropy(logits.float(), target, reduction="sum")

                    # At some points there is an inversion of the dimensions...
                    contrastive_loss_notReduced = nn.functional.cross_entropy(logits.float(), target,reduction="none")

                    if latent_time_reduction is not None:
                        x = contrastive_loss_notReduced.view(-1, mask_time_indices.shape[0])
                        contrastive_loss_notReduced = torch.stack([torch.sum(x[latent_time_reduction[i,:],i])/torch.sum(latent_time_reduction[i,:])
                                                                    for i in range(mask_time_indices.shape[0])],dim=0)
                        assert not torch.any(torch.isnan(contrastive_loss_notReduced))
                    else:
                        contrastive_loss_notReduced = torch.sum(
                            contrastive_loss_notReduced.view(-1, mask_time_indices.shape[0]), dim=0) / torch.sum(
                            mask_time_indices, dim=1)

                    # 7. compute diversity loss: \mathbf{L}_d
                    num_codevectors = self.config.num_codevectors_per_group * self.config.num_codevector_groups
                    diversity_loss = ((num_codevectors - codevector_perplexity) / num_codevectors) * mask_time_indices.sum()

                    # 8. \mathbf{L} = \mathbf{L}_m + \alpha * \mathbf{L}_d
                    loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss + features_pen

                return Wav2Vec2ForLossOutput(
                    loss=loss,
                    projected_states=transformer_features,
                    projected_quantized_states=quantized_features,
                    codevector_perplexity=codevector_perplexity,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                    contrastive_loss=contrastive_loss,
                    diversity_loss=diversity_loss,
                    pen_loss = features_pen,
                    pen_loss_nosum = features_pen/mask_time_indices.sum(),
                    contrastive_loss_notReduce = contrastive_loss_notReduced
                )

