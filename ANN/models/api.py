import abc
from abc import ABC, abstractmethod
from torch.nn import Module
import zarr
from typing import Union,List,Dict,Tuple
from pathlib import  Path
from datasets import DatasetDict,IterableDatasetDict,Dataset

class forPretraining(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self,input_values):
        pass

    @abstractmethod
    def save_loss(self,outputs : dict,zg: zarr.Group):
        pass
    @abstractmethod
    def get_mainloss_name(cls):
        pass

    @abstractmethod
    def load_config(cls,path : Union[str,Path]):
        pass

    @abstractmethod
    def from_pretrained(cls,path : Union[str,Path]):
        pass

    @abstractmethod
    def get_collator_pretraining(cls,file_configPreprocessor : Union[str,Path], config):
        # Given the configuration of the preprocessor, this methods should return the DataCollator
        # that is used during Training. As a reminder, the DataCollator batches several dict into a single dict.
        # which is then filtered of unused elements in the forward pass of the model and sent to the model as
        # argument to the forward function.
        pass

    @abstractmethod
    def pretransform_dataset(cls, ds: Union[DatasetDict,IterableDatasetDict],path: Union[str,Path]) -> Union[DatasetDict,IterableDatasetDict]:
        ## Some network needs to perform aditional pre-processing of the dataset
        # which we allow to be done in streaming by returning an IterableDatasetDict.
        pass

class forPostAnalysis(ABC,Module):
    @abstractmethod
    def layerzip(cls) -> List[Tuple[str,Dict]]:
        # Returns tuples of names of layer-block (like "convolutions"), [number of layers,layers width (number of unit)].
        # A layer-block is defined as a set of layers that returns vectors of equal dimensions.
        pass

    @abstractmethod
    def preprocessor_from_pretrained(self,path : Union[str,Path]):
        ### The idea is to have the same function for loading preprocessor
        # independently of the model
        pass

    @abstractmethod
    def get_collator(self,*kwargs):
        # This function should return a collator which is used to collate the data across batch
        # as well as adding additional tensor.
        pass

    @abstractmethod
    def preprocess_dataset(self,ds : Dataset) -> Dataset:
        # This function is run just prior to DataLoader(ds,...).
        # it is notably here that one should implement tokenization.
        pass

    @abstractmethod
    def get_downsampleSize(cls,input_size : int):
        # This function returns the number of element downsampled by the model in each of the layer
        # as a function of the size of the input.
        pass

class forLossAnalysis(forPostAnalysis):
    @abstractmethod
    def get_mainloss_name(cls):
        ## This function is used to select from all the output of the models
        # the ones that is really useful
        pass