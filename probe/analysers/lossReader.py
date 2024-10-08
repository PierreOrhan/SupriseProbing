import numpy as np
from ANN.models import POST_MODELS
from probe.analysers.utils import BaseBlockRepAnalyzer
from pathlib import Path

from torch.utils.data import DataLoader
import torch

import zarr as zr
import tqdm
from probe.analysers.utils import load_ANNdataset_withMask
import os

class PostLoss(BaseBlockRepAnalyzer):
    ## This class is designed for cluster usage

    def __init__(self, dataset_names, model_dir,data_dir,save_dir,
                 nb_total_checkpoint,chunk_size,path_preprocessor,model_type="wav2vec2",
                 causal=True):
        self.path_preprocessor = path_preprocessor
        super().__init__(dataset_names,model_dir,data_dir,save_dir,
                 nb_total_checkpoint,chunk_size,model_type,load=False)
        self.sound_Dataset = {}
        for d in dataset_names:
            self.sound_Dataset[d] = load_ANNdataset_withMask(Path(data_dir) / d,partially_causal=causal,extendWithMask=True)

    def _get_nbdownsample(self,sound_example,sampling_rate):
        ## make sure that all sound have the same length
        # The sounds waveform are assumed to be sampled at 16000 Hz
        assert len(np.unique([s.shape[-1] for s in sound_example])) == 1
        time_dim = int(np.floor(sampling_rate * sound_example[0].shape[-1] / 16000))
        return time_dim

    def alloc(self,chunk_sounds=False,nbchunks=200):
        #sampling_rate: used if average==False, in Hz: given the sound duration, parametrizes the number
        # of sound window we extract from of the network activity for each sound.

        zg = zr.open_group(os.path.join(self.save_dir,"postAnalyses_loss.zarr"),mode="a")
        for d in self.sound_Dataset.keys():
            nb_sounds,time_dim = self.sound_Dataset[d].info.dataset_resize
            # time_dim = 1
            chunks = (self.chunk_size, None, None, None, None)
            if chunk_sounds:
                chunks = (self.chunk_size, None, nbchunks, None, None)
            shape = (self.nb_checkpoint, 1, nb_sounds, time_dim, 1)
            if not d in zg.keys():
                zg.create(name=d, mode='a', shape=shape,
                            chunks = chunks, dtype = np.float32)

    def analyse(self,chunk_sounds=False,nbchunks=1, batch_size=4,remove_normalization : bool = False):
        # Load each checkpoint and save the loss on the dataset:
        zg = zr.open_group(os.path.join(self.save_dir, "postAnalyses_loss.zarr"), mode="a")
        # if not _is_initialized(zg):
        for id,chk_name in zip(self.checkpoints_id,self.checkpoints_names):
            model = POST_MODELS["loss"][self.model_type].from_pretrained(os.path.join(self.model_dir, chk_name))
            model.cuda()
            with torch.no_grad():
                for d in self.sound_Dataset.keys():
                    if np.all(np.equal(zg[d][id,0,-1,:,0],0)): # fast enough to test this...
                        # DataCollator from the dataset
                        data_collator = POST_MODELS["loss"][self.model_type].get_collator(
                            file_configPreprocessor=self.path_preprocessor,
                            remove_normalization= remove_normalization)
                        dataLoader = DataLoader(
                            self.sound_Dataset[d],
                            batch_size=batch_size,
                            collate_fn=data_collator,
                            shuffle=False,
                        )
                        ## Tips: although we use a IterableDataset we want to iterate for a finite number of time
                        ## calling for x in dataLoader would iterate infinitely...
                        itDataLoader = iter(dataLoader)

                        if nbchunks * self.sound_Dataset[d].info.dataset_resize[1] > self.sound_Dataset[
                            d].info.dataset_size:
                            nbchunks = self.sound_Dataset[d].info.dataset_resize[0]
                        totNbchunks = nbchunks*self.sound_Dataset[d].info.dataset_resize[1]

                        ## We force the number of chunks to be proportional to the number of
                        # elements in one sounds:

                        for idchk,chk_sound in tqdm.tqdm(enumerate(np.arange(self.sound_Dataset[d].info.dataset_size)[::totNbchunks])):
                            # if np.all(np.equal(zg[d + "_" + "trans"][id, -1, chk_sound:chk_sound + totNbchunks, ...],0)):
                            print("reading chunk sound", chk_sound)
                            def temp_itp():
                                # returns an iterator that yields nbChunks of sound vectors
                                # unless this is beyond the limit
                                if chk_sound > self.sound_Dataset[d].info.dataset_size:
                                    for _ in np.arange(self.sound_Dataset[d].info.dataset_size):
                                        yield next(itDataLoader)
                                else:
                                    max_lim = np.ceil((self.sound_Dataset[d].info.dataset_size - chk_sound) / batch_size)
                                    for _ in np.arange(np.min([np.ceil(totNbchunks / batch_size), max_lim]), dtype=int):
                                        yield next(itDataLoader)

                            mainloss_name = POST_MODELS["loss"][self.model_type].get_mainloss_name()
                            all_outputs = []
                            all_target = []
                            for idp, inputs in enumerate(temp_itp()):
                                outputs = model(**{inpk: inputs[inpk].to(model.device) for inpk in inputs.keys()})
                                idinputs = chk_sound + idp
                                target_index = np.unravel_index(
                                    np.arange(batch_size * idinputs,
                                              batch_size * idinputs + inputs["input_values"].shape[0]),
                                    # can't do (batch_size+1) as we might have less nb of inputs in the last batch
                                    self.sound_Dataset[d].info.dataset_resize)
                                all_outputs += [outputs[mainloss_name].detach().cpu().numpy()]
                                all_target += [target_index]
                            all_outputs = np.concatenate(all_outputs).reshape((min(nbchunks,self.sound_Dataset[d].info.dataset_resize[0]-idchk*nbchunks),-1))
                            zg[d][id, 0, idchk*nbchunks:idchk*nbchunks+nbchunks, :,0] = all_outputs

class PostLoss_perSound(BaseBlockRepAnalyzer):
    ## This class is designed for cluster usage

    def __init__(self, dataset_names, model_dir,data_dir,save_dir,
                 nb_total_checkpoint,chunk_size,path_preprocessor,model_type="wav2vec2",
                 causal=True):
        self.path_preprocessor = path_preprocessor
        super().__init__(dataset_names,model_dir,data_dir,save_dir,
                 nb_total_checkpoint,chunk_size,model_type,load=False)
        self.sound_Dataset = {}
        for d in dataset_names:
            self.sound_Dataset[d] = load_ANNdataset_withMask(Path(data_dir) / d,partially_causal=causal,extendWithMask=True)

    def _get_nbdownsample(self,sound_example,sampling_rate):
        ## make sure that all sound have the same length
        # The sounds waveform are assumed to be sampled at 16000 Hz
        assert len(np.unique([s.shape[-1] for s in sound_example])) == 1
        time_dim = int(np.floor(sampling_rate * sound_example[0].shape[-1] / 16000))
        return time_dim

    def alloc(self):
        zg = zr.open_group(os.path.join(self.save_dir,"postAnalyses_loss.zarr"),mode="a")
        ## We generate one array per output sound.
        for d in self.sound_Dataset.keys():
            example_sd = next(iter(self.sound_Dataset[d]))
            itp = iter(self.sound_Dataset[d])
            for idsd in range(self.sound_Dataset[d].info.dataset_nbsound):
                if "latent_time_reduction" in example_sd.keys():
                    if isinstance(self.sound_Dataset[d].info.nb_element,int):
                        time_dim = self.sound_Dataset[d].info.nb_element
                    else:
                        time_dim = self.sound_Dataset[d].info.nb_element[idsd]
                        # double check no problem here (thinking of shuffling issue..)
                else:
                    sd = next(itp)
                    time_dim = POST_MODELS["loss"][self.model_type].get_downsampleSize(sd["input_size"])
                    ## iterate across all elements coming from the same sound:
                    for _ in range(self.sound_Dataset[d].info.nb_element[idsd]):
                        x = next(itp)

                # chunks = (self.chunk_size, 1, None, None)
                chunks = (self.chunk_size, None, None,None)
                shape = (self.nb_checkpoint, 1, time_dim, 1)
                if not d+"_"+self.sound_Dataset[d].info.names[idsd] in zg.keys():
                    zg.create(name=d+"_"+self.sound_Dataset[d].info.names[idsd], shape=shape,
                                chunks = chunks, dtype = np.float32)

    def analyse(self, batch_size=1,remove_normalization : bool = False):

        ### TODO: improve by batching probing of the same sound, which would generate batch of different size...

        assert batch_size ==1
        # Load each checkpoint and save the loss on the dataset:
        zg = zr.open_group(os.path.join(self.save_dir, "postAnalyses_loss.zarr"), mode="a")
        # if not _is_initialized(zg):
        for id,chk_name in zip(self.checkpoints_id,self.checkpoints_names):
            model = POST_MODELS["loss"][self.model_type].from_pretrained(os.path.join(self.model_dir, chk_name))
            model.cuda()
            with torch.no_grad():
                for d in self.sound_Dataset.keys():
                    # DataCollator from the dataset
                    data_collator = POST_MODELS["loss"][self.model_type].get_collator(
                        file_configPreprocessor=self.path_preprocessor,
                        remove_normalization= remove_normalization)
                    dataLoader = DataLoader(
                        self.sound_Dataset[d],
                        batch_size=batch_size,
                        collate_fn=data_collator,
                        shuffle=False,
                    )

                    ## Tips: although we use a IterableDataset we want to iterate for a finite number of time
                    ## calling for x in dataLoader would iterate infinitely...
                    itDataLoader = iter(dataLoader)
                    mainloss_name = POST_MODELS["loss"][self.model_type].get_mainloss_name()
                    for idp in np.arange(self.sound_Dataset[d].info.dataset_nbsound):
                        all_res = []
                        for element_id in tqdm.tqdm(np.arange(self.sound_Dataset[d].info.nb_element[idp])):
                            inputs = next(itDataLoader)
                            outputs = model(**{inpk: inputs[inpk].to(model.device) for inpk in inputs.keys()})
                            all_res += [outputs[mainloss_name].detach().cpu().numpy()]
                        zg[d+ "_" +self.sound_Dataset[d].info.names[idp]][id, 0, :,0] = np.concatenate(all_res)

    def analyse_sess(self,datasetkey,id_sound,batch_size=1,remove_normalization : bool = False):
        """
        :param datasetkey: the key of the subdataset
        :param id_sound: the id of the sound to explore...
        :param batch_size:
        :param remove_normalization:
        :return:
        """
        ### TODO: improve by batching probing of the same sound, which would generate batch of different size...
        assert batch_size ==1
        # Load each checkpoint and save the loss on the dataset:
        zg = zr.open_group(os.path.join(self.save_dir, "postAnalyses_loss.zarr"), mode="a")
        # if not _is_initialized(zg):
        for id,chk_name in zip(self.checkpoints_id,self.checkpoints_names):
            model = POST_MODELS["loss"][self.model_type].from_pretrained(os.path.join(self.model_dir, chk_name))
            model.cuda()
            with torch.no_grad():
                data_collator = POST_MODELS["loss"][self.model_type].get_collator(
                    file_configPreprocessor=self.path_preprocessor,
                    remove_normalization= remove_normalization)
                dataLoader = DataLoader(
                    self.sound_Dataset[datasetkey],
                    batch_size=batch_size,
                    collate_fn=data_collator,
                    shuffle=False,
                )
                ### TODO: improve so that we don't have to iter through the whole dataset...

                ## Tips: although we use a IterableDataset we want to iterate for a finite number of time
                ## calling for x in dataLoader would iterate infinitely...
                itDataLoader = iter(dataLoader)
                mainloss_name = POST_MODELS["loss"][self.model_type].get_mainloss_name()
                for idp in np.arange(id_sound+1):
                    if id_sound==idp:
                        all_res = []
                        for element_id in np.arange(self.sound_Dataset[datasetkey].info.nb_element[idp]):
                            inputs = next(itDataLoader)
                            outputs = model(**{inpk: inputs[inpk].to(model.device) for inpk in inputs.keys()})
                            all_res += [outputs[mainloss_name].detach().cpu().numpy()]
                        zg[datasetkey+ "_" +self.sound_Dataset[datasetkey].info.names[idp]][id, 0, :,0] = np.concatenate(all_res)
                    else:
                        for element_id in np.arange(self.sound_Dataset[datasetkey].info.nb_element[idp]):
                            inputs = next(itDataLoader)