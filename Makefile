# Bash shell needed for source function in conda
SHELL=/usr/bin/env bash

# NOTE: The | for prerequisites above makes sure that it ignores the timestamp of the folders
INSTALL_DIR=/local/scratch/a/shared/starcraft_shared
CONDA_INSTALL_DIR=/local/scratch/a/$(USER)

dummy:
	# Use `make install` to install

install: maps replays | $(CONDA_INSTALL_DIR)/miniconda3/envs/starcraft data $(INSTALL_DIR)/StarCraftII 
	# !!!! Make sure to update your .bashrc file according to the README
	# and then source your .bashrc file for the environment to take effect!!!!!

data:
	mkdir data

sensor-dataset: | data/starcraft-sensor-dataset
	# Alias

data/starcraft-sensor-dataset: | data
	wget -O $(@D)/$(@F).zip https://figshare.com/ndownloader/files/35923727?private_link=b56ef1c8cc8c87e9115f
	pushd $(@D); unzip $(@F).zip; popd
	rm $(@D)/$(@F).zip



$(INSTALL_DIR):
	mkdir $(INSTALL_DIR)
	-chmod -R g+rwx $(INSTALL_DIR)

$(CONDA_INSTALL_DIR)/miniconda3:
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	bash Miniconda3-latest-Linux-x86_64.sh -b -p $(CONDA_INSTALL_DIR)/miniconda3
	rm Miniconda3-latest-Linux-x86_64.sh

$(CONDA_INSTALL_DIR)/miniconda3/envs/starcraft: $(CONDA_INSTALL_DIR)/miniconda3
	# Setup conda packages
	#$(CONDA_INSTALL_DIR)/miniconda3/bin/conda create -y -n starcraft python=3.7 pytorch torchvision cudatoolkit=11.3 scikit-image=0.19 scipy>=1.4.1 -c pytorch
	# Add required pip packages
	#source $(CONDA_INSTALL_DIR)/miniconda3/bin/activate starcraft && pip install sparse==0.13 --upgrade pysc2
	$(CONDA_INSTALL_DIR)/miniconda3/bin/conda env create -f environment-simple.yml

$(INSTALL_DIR)/StarCraftII: | $(INSTALL_DIR)
	wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.3.16.1.zip
	unzip -P iagreetotheeula SC2.3.16.1.zip -d $(INSTALL_DIR)
	rm SC2.3.16.1.zip
	-chmod -R g+rwx $(INSTALL_DIR)

maps: | $(INSTALL_DIR)/StarCraftII/Maps/Ladder2017Season1 $(INSTALL_DIR)/StarCraftII/Maps/Melee
	-chmod -R g+rwx $(INSTALL_DIR)

$(INSTALL_DIR)/StarCraftII/Maps/%: | $(INSTALL_DIR)/StarCraftII
	wget http://blzdistsc2-a.akamaihd.net/MapPacks/$(@F).zip -O $(INSTALL_DIR)/StarCraftII/Maps/$(@F).zip
	unzip -P iagreetotheeula $(INSTALL_DIR)/StarCraftII/Maps/$(@F).zip -d $(INSTALL_DIR)/StarCraftII/Maps/
	rm $(INSTALL_DIR)/StarCraftII/Maps/$(@F).zip

replays: | $(INSTALL_DIR)/StarCraftII/Replays-3.16.1-Pack_1-fix
	-chmod -R g+rwx $(INSTALL_DIR)

$(INSTALL_DIR)/StarCraftII/Replays-%: | $(INSTALL_DIR)/StarCraftII
	wget http://blzdistsc2-a.akamaihd.net/ReplayPacks/$*.zip -O $(INSTALL_DIR)/StarCraftII/$*.zip
	unzip -P iagreetotheeula $(INSTALL_DIR)/StarCraftII/$*.zip -d $(INSTALL_DIR)/StarCraftII/
	# Move to its own replay directory
	mv $(INSTALL_DIR)/StarCraftII/Replays $(INSTALL_DIR)/StarCraftII/Replays-$*
	rm $(INSTALL_DIR)/StarCraftII/$*.zip

.PHONY: clean-all clean-miniconda3 clean-maps clean-replays clean-StarCraftII debug replay play environment export-environment

clean-all: clean-StarCraftII clean-pysc2 clean-iniconda3

clean-environment:
	conda env remove --name starcraft

clean-miniconda3:
	rm -rf $(CONDA_INSTALL_DIR)/miniconda3

clean-StarCraftII:
	rm -r $(INSTALL_DIR)/StarCraftII

clean-maps:
	rm -r $(INSTALL_DIR)/StarCraftII/Maps/*

clean-replays:
	rm -r $(INSTALL_DIR)/StarCraftII/Replays*

environment: $(CONDA_INSTALL_DIR)/miniconda3/envs/starcraft
	# Alias

export-environment:
	# Using --from-history is important as it only keeps the explicitly requested packages
	conda env export --from-history

replay:
	python script-extract-data.py --feature_minimap_size 0 --rgb_screen_size 0 --replay $(INSTALL_DIR)/StarCraftII/Replays-3.16.1-Pack_1-fix/bfed1a7e1daea8ca1447ca39f2d144d57acebbcb3cd011ba2662d313ba450178.SC2Replay

debugshort:
	python -m pdb script-extract-data.py --max_steps=300 --feature_minimap_size 0 --rgb_screen_size 0 --replay $(INSTALL_DIR)/StarCraftII/Replays-3.16.1-Pack_1-fix/bfed1a7e1daea8ca1447ca39f2d144d57acebbcb3cd011ba2662d313ba450178.SC2Replay

debug:
	# Gives 0 key error
	python -m pdb script-extract-data.py --feature_minimap_size 0 --rgb_screen_size 0 --replay $(INSTALL_DIR)/StarCraftII/Replays-3.16.1-Pack_1-fix/536f520aef4f3ab76c70e298b6c0bf6801316a53628d422396f01e40fb30a110.SC2Replay

killsc:
	-pkill --full SC2_x64

dataset: killsc
	# Now run the extraction
	python script-extract-data.py --num_parallel_jobs 40 \
	--replay_dir $(INSTALL_DIR)/StarCraftII/Replays-3.16.1-Pack_1-fix
	--data_save_dir data/starcraft-sensor-dataset

parallel1k:
	python script-extract-data.py --max_replays 1000 --num_parallel_jobs 40 --feature_minimap_size 0 --rgb_screen_size 0 --replay_dir $(INSTALL_DIR)/StarCraftII/Replays

parallelshort:
	python script-extract-data.py --max_replays 2 --num_parallel_jobs 2 --feature_minimap_size 0 --rgb_screen_size 0 --replay_dir $(INSTALL_DIR)/StarCraftII/Replays

play:
	python play.py --map Simple64
