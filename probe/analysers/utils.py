import os

import numpy as np
from probe.api import PostAnalyser
from datasets import IterableDataset,Dataset
import zarr as zr
from pathlib import Path
import pandas as pd
import soundfile as sf
from typing import Union

def load_ANNdataset(dataset_dir : Path) -> Union[IterableDataset,Dataset]:
    """
    Converts an ANNdataset to a torch dataset, usable in pytorch.
    :param dataset_dir: directory toward a dataset as formatted by ControlledStim package.
    :return:
        Huggingface's datasets IterableDataset, with dataset.info.dataset_size filed to allow it to be used by DataLoader.
    """
    sequences = pd.read_csv(dataset_dir / "trials.csv")

    ## TODO: implement a way to get what the format of the dataset is...
    if "wav_path" in sequences.keys():
        def gen(shards):
            for shard in shards:
                sequence = sequences.iloc[shard, :]
                sd,sr = sf.read(sequence["wav_path"])
                yield {"input_values":sd,"input_size":sd.shape[-1],"sr":sr}

        shards = np.arange(sequences.shape[0])
        ds = IterableDataset.from_generator(gen, gen_kwargs={"shards": shards})
        ds = ds.with_format("torch")
        # ---> Transforms to a TorchIterableDataset (named IterableDataset in pytorch)
        # which has the attribute len and can be used
        # in a DataLoader (i.e combined with a collator!!)
        ds.info.dataset_size = sequences.shape[0]
    else:
        os.makedirs(str(dataset_dir/".cache"),exist_ok=True)
        ds = Dataset.from_csv(str(dataset_dir / "trials.csv"),cache_dir=str(dataset_dir/".cache"))
        ds = ds.with_format(None, columns=["sentences"], output_all_columns=False)
        ds.info.dataset_size = ds.num_rows

        ## isalpha(): We remove from the data all elements that are entirely composed of non-letters!
        # ie we do not predict independent punctuations or words.
        ds = ds.map(lambda x:{"input_size":len(list(filter(lambda e: np.any([b.isalpha() or b.isdigit() or b=="-" for b in e]),
                                                    x["sentences"].split(" "))))},batched=False)

    return ds

### Observation: we should be able to provide the same reader but for text data....

def load_ANNdataset_withMask(dataset_dir : Path,partially_causal = True,extendWithMask=True) -> IterableDataset:
    """
    Converts an ANNdataset to a torch dataset, usable in pytorch.
    TODO: Fix the fact that this loading method assume sounds of constant size, which facilitates the work of the analyser method.
    :param dataset_dir: directory toward a dataset as formatted by ControlledStim package.
    :param partially_causal: adds a latent_attention_mask (vector) to the model inputs, which can be used to generate
    a mask at the level of attention to prevent the attention on future token, be careful to also use it to set to 0 any
    positional encoding that would use local convolution.
    :param extendWithMask: if we yield the sound for every different masks associated to the sounds
    :return:
        Huggingface's datasets IterableDataset, with dataset.info.dataset_size filed to allow it to be used by DataLoader.
    """
    sequences = pd.read_csv(dataset_dir / "trials.csv")
    def gen(shards):
        for shard in shards:
            sequence = sequences.iloc[shard, :]
            sd,sr = sf.read(sequence["wav_path"])
            zg = zr.open_group(sequence["mask_info_path"],mode="r")

            if "latent_time_reduction" in zg.keys():
                ltr = zg["latent_time_reduction"]
                has_ltr = True
            else:
                has_ltr = False

            if extendWithMask:
                mti = zg["mask_time_indices"]
                sni = zg["sampled_negative_indices"]
                for element_id in range(zg["mask_time_indices"].shape[0]):
                    if partially_causal:
                        ## Adds attention masking, to forbid the use of future elements:
                        attention_mask = np.zeros(mti.shape[-1],dtype=bool) + True
                        end_mask = np.where(mti[element_id,...])[0][-1]
                        attention_mask[end_mask+1:] = False

                        if has_ltr:
                            yield {"input_values": sd, "mask_time_indices": mti[element_id, ...],
                                   "sampled_negative_indices": sni[element_id, ...],
                                   "latent_attention_mask":attention_mask,
                                   "latent_time_reduction":ltr[element_id,...]}
                        else:
                            yield {"input_values": sd, "mask_time_indices": mti[element_id, ...],
                                   "sampled_negative_indices": sni[element_id, ...],
                                   "latent_attention_mask":attention_mask}
                    else:
                        if has_ltr:
                            yield {"input_values": sd,
                                   "mask_time_indices": mti[element_id, ...],
                                   "sampled_negative_indices": sni[element_id, ...],
                                   "latent_time_reduction":ltr[element_id,...]}
                        else:
                            yield {"input_values": sd,
                                   "mask_time_indices": mti[element_id, ...],
                                   "sampled_negative_indices": sni[element_id, ...]}
            else:
                if partially_causal:
                    raise Exception("causal reading of the activity without providing the mask is not implemented")
                if has_ltr:
                    yield {"input_values": sd,"latent_time_reduction":ltr}
                else:
                    yield {"input_values":sd}

    shards = np.arange(sequences.shape[0])
    ds = IterableDataset.from_generator(gen, gen_kwargs={"shards": shards})
    ds = ds.with_format("torch")
    # ---> Transforms to a TorchIterableDataset (named IterableDataset in pytorch)
    # which has the attribute len and can be used
    # in a DataLoader (i.e combined with a collator!!)

    if extendWithMask:
        ds.info.dataset_size = np.sum(sequences["number_element"])
        ds.info.dataset_nbsound = sequences.shape[0]
        ds.info.nb_element = sequences["number_element"]
        ds.info.names = sequences["name"].values
    else:
        ds.info.dataset_size = sequences.shape[0]
        if "number_element" in sequences.keys():
            if np.unique(sequences["number_element"]).shape[0] == 1:
                ds.info.nb_element = sequences["number_element"][0]
            else:
                ds.info.nb_element = sequences["number_element"]
        ds.info.names = sequences["name"].values
    return ds
#
def load_ANNdataset_causalWithoutMask(dataset_dir : Path) -> IterableDataset:
    """
    Converts an ANNdataset to a torch dataset, usable in pytorch.
    This loader does not output a "mask_time_indices" such that we can probe how the model's activity of a surprising
    tone emerge, while still using partial causality.
    The loader ouputs nb_element sound for every sound in the dataset, each containing nb_element.
    TODO: Fix the fact that this loading method assume sounds of constant size, which facilitates the work of the analyser method.
    :param dataset_dir: directory toward a dataset as formatted by ControlledStim package.
    :return:
        Huggingface's datasets IterableDataset, with dataset.info.dataset_size filed to allow it to be used by DataLoader.
    """
    sequences = pd.read_csv(dataset_dir / "trials.csv")

    def gen(shards):
        for shard in shards:
            sequence = sequences.iloc[shard, :]
            sd, sr = sf.read(sequence["wav_path"])
            zg = zr.open_group(sequence["mask_info_path"], mode="r")

            if "latent_time_reduction" in zg.keys():
                ltr = zg["latent_time_reduction"]
                has_ltr = True
            else:
                has_ltr = False

            mti = zg["mask_time_indices"]
            for element_id in range(zg["mask_time_indices"].shape[0]):
                ## Adds attention masking, to forbid the use of future elements:
                attention_mask = np.zeros(mti.shape[-1], dtype=bool) + True
                end_mask = np.where(mti[element_id, ...])[0][-1]
                attention_mask[end_mask + 1:] = False

                if has_ltr:
                    yield {"input_values": sd,
                           "latent_attention_mask": attention_mask,
                           "latent_time_reduction": ltr[element_id, ...]}
                else:
                    yield {"input_values": sd,
                           "latent_attention_mask": attention_mask}

    shards = np.arange(sequences.shape[0])
    ds = IterableDataset.from_generator(gen, gen_kwargs={"shards": shards})
    ds = ds.with_format("torch")
    # ---> Transforms to a TorchIterableDataset (named IterableDataset in pytorch)
    # which has the attribute len and can be used
    # in a DataLoader (i.e combined with a collator!!)
    ds.info.dataset_size = np.sum(sequences["number_element"])
    assert np.unique(sequences["number_element"]).shape[0] == 1
    ds.info.dataset_resize = (sequences.shape[0], sequences["number_element"][0])
    return ds



class BaseBlockRepAnalyzer(PostAnalyser):
     # simply to avoir having these lines wrote inside each Analyzer
    def __init__(self,dataset_names,model_dir,data_dir,save_dir,
                 nb_total_checkpoint,chunk_size,model_type="wav2vec2",load=True):

        from numcodecs import blosc
        blosc.use_threads = False  # safe usage of Zarr's compressor Blosc in multi-process environment

        # Read and prepare the sound files in cuda base on dataset_name
        # Prepare the zarr files to store the results of the analysis
        # Prepare the masked dataset
        self.nb_checkpoint = nb_total_checkpoint
        self.chunk_size = chunk_size
        self.save_dir = save_dir
        self.model_dir = model_dir
        self.model_type = model_type
        self.data_dir = data_dir

        self.sound_tensor = {}
        self.sound_Dataset = {}

    def alloc(self,*kwargs):
        raise Exception("Not implemented")
    def analyse(self, *kwargs):
         raise Exception("Not implemented")

    def multiprocessinit(self,checkpoints_id,checkpoints_names):
        # Set the checkpoints used by this analyser
        self.checkpoints_id = checkpoints_id
        self.checkpoints_names = checkpoints_names


def _is_initialized(zg):
    initialized  = []
    for k in zg.keys():
        try:
            initialized += [zg[k].nchunks == zg[k].nchunks_initialized]
        except:
            initialized += [False]
    if len(initialized)==0:
        return False
    return np.all(np.array(initialized))
