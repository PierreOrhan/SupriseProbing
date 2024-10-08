from abc import ABC, abstractmethod

class PostAnalyser(ABC):
    # The PostAnalyser provide API to load checkpoints and perform some analysis of the models

    @abstractmethod
    def alloc(self,*kwargs):
        # Should allocate the zarr chunks before the analysis
        pass

    @abstractmethod
    def analyse(self,*kwargs):
        # Should perform the analysis
        pass

    @abstractmethod
    def multiprocessinit(self,checkpoints_id,checkpoints_names):
        pass