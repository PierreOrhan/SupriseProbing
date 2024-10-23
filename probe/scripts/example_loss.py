## This script allows you to plot the model surprise for one protocol

from  ANN.models import  Wav2vec2_forLoss_ConstrainedMask
import tqdm
from torch.utils.data import DataLoader
from probe.analysers.utils import  load_ANNdataset_withMask
import pathlib
from pathlib import Path
import matplotlib.pyplot as plt

## To use this script one need to have generated a dataset from ControlledStim.


data_dir = "/yourdirtothedata/lot_testGeneration/test_randregrand"
ds = load_ANNdataset_withMask(pathlib.Path(data_dir))
ds = ds.with_format("torch")
#---> Transforms to a TorchIterableDataset which has the attribute len and can be used
# in a DataLoader (i.e combined with a collator!!)

## Models can be downloaded here: https://huggingface.co/NDEM
dir_model = "/yourdirtothemodel/outputs_mergefilter_short_2/checkpoint-100000"
model = Wav2vec2_forLoss_ConstrainedMask.from_pretrained(dir_model)
model.save_pretrained("../.." / pathlib.Path(data_dir))

path_config =  Path("/home/pierre/Documents/NEM/ANN/models/wav2vec2/config")
path_preprocessor = path_config / "preprocessor_config.json"


batch_size = 4
data_collator = Wav2vec2_forLoss_ConstrainedMask.get_collator(file_configPreprocessor= path_preprocessor)
dataLoader = DataLoader(
    ds,
    batch_size=batch_size,
    collate_fn=data_collator,
    shuffle=False,
)

all_loss = []
for inputs in tqdm.tqdm(dataLoader):

    print("reading input")
    outputs = model(**{inpk: inputs[inpk].to(model.device) for inpk in inputs.keys()})
    mainloss_name = Wav2vec2_forLoss_ConstrainedMask.get_mainloss_name()

    print("saving loss")
    res = outputs[mainloss_name].detach().cpu().numpy()
    all_loss += [r for r in res]

fig,ax = plt.subplots()
ax.plot(all_loss)
fig.show()
