# StarCraft II Dataset Extractor

Welcome! This is the [anonymous] repository for the StarCraftImage Dataset.

## Downloading the Code:

If you are seeing this on the Open4Science Repo, unfortunately, there is not an easy way to download a full repository via Open4Science, so to clone/download this repository, please see the mirroring GitHub page: https://github.com/anonymous-researcher-42/starcraft-dataset-extraction-private

If on the GitHub repo, one can just clone the repo as usual, and then to call `$ make dataset` to download and extract the dataset.

## Dataset Nutrition Label
![](StarCraftImage-Nutrition-Label-cropped.png)

## Downloading the dataset
This can be done by either manually installing from this link: https://figshare.com/s/b56ef1c8cc8c87e9115f

Or one can call `$ make dataset` to download and extract the dataset automatically.  [Note: the command must be executed while inside this repository directory]

From here you can either directly load the StarCraft{MNIST, CIFAR10}_{train, test}.npz files and work with those as if you were working with MNIST or CIFAR10 itself. Additionally, you can use `sc2image.dataset` class to interact with the data and metadata more directly (please see the demo notebooks below for examples).

## Running demos
To rerun the demos, first replicate the python environment seen in the `environment.yml` file   (which can be done manually, or by calling `$ make environment` inside this repository).

Then follow the steps in the **Downloading the dataset section** above to download the StarCraftImage dataset.

Lastly, you will need to update the `root_dir` and `subdir` paths seen in the jupyter notebooks to match where you extracted the dataset, and then you should be able to run the notebooks!


## Replicating the extraction
If you would like to replicate the extraction process, you first will need to install the proper StarCraft II game files, replay files, and pysc2 files. We've mostly automated this process for you, so to setup this repository please run the following commands:

`$ make install`   which will should install the maps, replays, and game files for StarCraft II to run on.

`$ make environment`  can **optionally** be called to make a miniconda environment named `starcraft` which will be built from the environment.yml file.

After this is done, please add the following variables to your .bashrc or similar file with the following (or put in a separate file and use source <YOUR_FILE> so that the variables are exported in your current environment):

```
# Setup StarCraftAI environment variables
INSTALL_DIR=.
CONDA_INSTALL_DIR=.
export SC2PATH=${INSTALL_DIR}/StarCraftII/
eval "$(${CONDA_INSTALL_DIR}/miniconda3/bin/conda shell.bash hook)"  # optional addition
conda activate starcraft  # optional addition
```

From here, you can just call:
`$ make dataset`, and the extraction should begin. 
Note: this will replicate the full 30k dataset extraction (which for reference should around a week to run on a 24 core machine). 
Instead, you could also call `$ make 1k-dataset` which will extract a smaller dataset of only one thousand replays (i.e. 1/30 of the full dataset size).
