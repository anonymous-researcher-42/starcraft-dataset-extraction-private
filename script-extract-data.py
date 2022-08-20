#!/usr/bin/python
"""Run SC2 to play a game or a replay."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import subprocess
import sys
import os
import time
from warnings import warn
from pathlib import Path
import json

import logging
import enum

import numpy as np
import joblib

from absl import app
from absl import flags
from pysc2 import run_configs
from pysc2.lib import point_flag
from pysc2.lib import replay
from pysc2.lib import stopwatch

from s2clientprotocol import sc2api_pb2 as sc_pb

# Extractor
from sc2sensor.extract import DataExtractor

FLAGS = flags.FLAGS

# Our extractor-specific flags
flags.DEFINE_string("replay_dir", None, "Directory of replays.")
flags.DEFINE_string("replay", None, "Single replay file.")
flags.DEFINE_integer("max_replays", -1,"Maximum number of replays to process")
flags.DEFINE_integer("num_parallel_jobs", 1, "Number of parallel jobs on CPU")
point_flag.DEFINE_point("raw_image_size", "64", "Extracted image size.")
flags.DEFINE_string("data_save_dir", f"data/dataset-extraction-{time.strftime('%Y_%m_%d_%H%M%S')}",
                     "Top-level directory to save extracted dataset.")
flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("summarize_mul", 255, "Number of frames to summarize over.")

# Other flags from pysc2 play.py file (retained to be backwards compatible)
flags.DEFINE_integer("step_mul", 1, "Game steps per observation.")
flags.DEFINE_integer("max_steps", -1, "Maximum number of steps to process")

# Feature sizes
point_flag.DEFINE_point("feature_screen_size", "0",
                        "Resolution for screen feature layers.")
point_flag.DEFINE_point("feature_minimap_size", "0",
                        "Resolution for minimap feature layers.")
flags.DEFINE_integer("feature_camera_width", 0,
                     "Width of the feature layer camera.")
point_flag.DEFINE_point("rgb_screen_size", "0",
                        "Resolution for screen feature layers.")
point_flag.DEFINE_point("rgb_minimap_size", "0",
                        "Resolution for minimap feature layers.")


flags.DEFINE_string("map_path", None, "Override the map for this replay.")

# Not sure if these are needed
flags.DEFINE_bool("full_screen", False, "Whether to run full screen.")
point_flag.DEFINE_point("window_size", "640,480",
                        "Screen size if not full screen.")


# initializing logger
logger = logging.getLogger('ds_extraction')
logger.setLevel(logging.DEBUG)
logger.propagate = False
# the rest of the logger will be set up in main() after FLAGS has been parsed


def get_replays_to_do():
    # Get files in replay folder
    files = [os.path.join(FLAGS.replay_dir, f)
             for f in os.listdir(FLAGS.replay_dir) 
             if os.path.isfile(os.path.join(FLAGS.replay_dir, f))]
    not_replays = [f for f in files if not f.lower().endswith("sc2replay")]
    if len(not_replays) > 0:
        logger.warning(f'Skipping the following non-replay files:\n{not_replays}')
    replay_file_list = [f for f in files if f.lower().endswith("sc2replay")]

    return replay_file_list



def kill_controller(controller, timeout=5):
    client = controller._client
    # recreating the controller.close() functionality to make sure we properly kill the process
    if client._sock:
        client._sock.close()
        # client._sock = None
        time.sleep(1)
    # Create a python version of the Status enum in the proto.
    Status = enum.Enum("Status", sc_pb.Status.items())
    client._status = Status.quit
    return None

def make_run_config(flags, replay_file_list):
    # FLAGS info
    def serialize_script_flags(flags):
        key_flags = flags.get_key_flags_for_module(sys.argv[0])
        return dict(
            flags={f.name:f.value for f in key_flags},
            sys_argv=sys.argv,
            original_command=' '.join(sys.argv),
        )
    # GIT info
    def get_output(command_str, command_label):
        try:
            out = subprocess.check_output(command_str.split(' ')).decode('ascii').strip()
        except subprocess.CalledProcessError as e:
            logger.warning(f'Could not extract {command_label} for metadata')
            out = '{command_label}_unknown'
        return out

    run_config = dict(replay_files = replay_file_list,
                      script_info = serialize_script_flags(flags),
                      git_hash = get_output('git rev-parse HEAD', 'git_hash'),
                      git_diff = get_output('git diff HEAD --no-color', 'git_diff'),
                      git_untracked = get_output('git ls-files --others --exclude-standard', 'git_untracked'))
    return run_config

def initiate_start_replay(FLAGS, replay_data, observed_player_id=1, disable_fog=False):
    """Setups up a SC_PB interface, initiates a request to start the replay, and returns
    the output of the StartReplayRequest"""
    # Setup interface (this is key for what is available in the obs object)
    interface = sc_pb.InterfaceOptions()
    # Important to get raw data out of observation
    # Try to make it get as much visibility as possible
    interface.raw = True 
    interface.raw_affects_selection = False
    interface.raw_crop_to_playable_area = False
    interface.score = True
    interface.show_cloaked = True 
    interface.show_burrowed_shadows = True
    interface.show_placeholders = True
    if FLAGS.feature_screen_size and FLAGS.feature_minimap_size:
        raise RuntimeError('feature_screen_size and feature_minimap_size should no longer be used to save on observation time')
        #interface.feature_layer.width = FLAGS.feature_camera_width
        #FLAGS.feature_screen_size.assign_to(interface.feature_layer.resolution)
        #FLAGS.feature_minimap_size.assign_to(
        #    interface.feature_layer.minimap_resolution)
        #interface.feature_layer.crop_to_playable_area = False
        #interface.feature_layer.allow_cheating_layers = True

    start_replay = sc_pb.RequestStartReplay(replay_data=replay_data,
                                            options=interface,
                                            disable_fog=disable_fog,
                                            observed_player_id=observed_player_id)
    return start_replay

def extract_just_player_info(extractor, replay_file, player_id):
    assert player_id in [1, 2], f'Player id must be either 1 or 2, got {player_id}.'
    # Load Replay
    run_config = run_configs.get()
    replay_data = run_config.replay_data(replay_file)
    version = replay.get_replay_version(replay_data)
    run_config = run_configs.get(version=version)  # Replace the run config.
    # initiate the replay, but from player_id's perspective now
    start_replay = initiate_start_replay(FLAGS, replay_data, observed_player_id=player_id)
    # Connect to engine and start the replay
    logger.debug(f'\nStarting player {player_id} data extraction...')
    with run_config.start(want_rgb=False) as p_controller:
        info = p_controller.replay_info(replay_data)
        map_path = FLAGS.map_path or info.local_map_path
        if map_path:
            start_replay.map_data = run_config.map_data(map_path,
                                                        len(info.player_info))
        p_controller.start_replay(start_replay)

        try:
            step = 0 # Keep track of steps
            start_time = time.time()
            while True:
                p_controller.step(FLAGS.step_mul)
                step += FLAGS.step_mul

                obs = p_controller.observe()
                extractor.extract_and_append_just_player_data(obs, step, player_id)

                # Show a little debug info on steps
                if step % 1000 == 0:
                    current_time = time.time() - start_time
                    start_time = time.time()
                    logger.debug(f'Player {player_id} Extraction: Step: {step} Time for steps: {current_time}')
                
                if obs.player_result or (FLAGS.max_steps != -1 and step >= FLAGS.max_steps):
                    return True

        except KeyboardInterrupt:
            return False

        except Exception as e:
            # some error during the two player extraction has occurred
            log_message = f'\n\nError in replay player {player_id} extraction.\nReplay: {str(replay_file)}\n{str(e)}'
            logger.error(log_message)
            return False

        finally:
            # NOTE: this will **always** be called, regardless of if a return is encountered in try or except
            #       and will return whatever return value was set in try (which is return=True) or except (re: False)
            kill_controller(p_controller)


def extract_replay(replay_file):
  """Run SC2 to play a game or a replay."""
  logger.info(f'Starting {replay_file}')
  FLAGS.replay = replay_file # Hacky way to save replay file for extract

  if FLAGS.trace:
    stopwatch.sw.trace()
  elif FLAGS.profile:
    stopwatch.sw.enable()

  if not replay_file:
    sys.exit("Must supply replay file.")

  if replay_file and not replay_file.lower().endswith("sc2replay"):
    sys.exit("Replay must end in .SC2Replay.")

  # Load Replay
  run_config = run_configs.get()
  replay_data = run_config.replay_data(replay_file)
  version = replay.get_replay_version(replay_data)
  run_config = run_configs.get(version=version)  # Replace the run config.
 
  start_replay = initiate_start_replay(FLAGS, replay_data, observed_player_id=1, disable_fog=True)

  # Connect to engine and start the replay
  with run_config.start(want_rgb=False) as controller:
    info = controller.replay_info(replay_data)

    # checking if replay is long enough to extract data from
    if  info.game_duration_loops < 2*FLAGS.summarize_mul:
      logger.error(f'Skipping replay {replay_file} as it is too short with {info.game_duration_loops} frames ' +
                     f'and a window size of {FLAGS.summarize_mul}.')
      return None

    logger.debug(" Replay info ".center(60, "-")+str(info)+"-" * 60)
    map_path = FLAGS.map_path or info.local_map_path
    if map_path:
      start_replay.map_data = run_config.map_data(map_path,
                                                  len(info.player_info))
    controller.start_replay(start_replay)


    try:
      # Setup data to collect through loop
      extractor = DataExtractor(game_info=controller.game_info(),
                                sc2_flags=FLAGS,
                                replay_info=info)

      step = 0 # Keep track of steps
      start_time = time.time()
      while True:
        controller.step(FLAGS.step_mul)
        step += FLAGS.step_mul

        obs = controller.observe()
        extractor.extract_and_append_data(obs, step)

        # Show a little debug info on steps
        if step % 1000 == 0:
          current_time = time.time() - start_time
          start_time = time.time()
          unique_players = np.unique([u.owner for u in obs.observation.raw_data.units])
          unit_counts = {
              player_id: len([u for u in obs.observation.raw_data.units if u.owner == player_id])
              for player_id in unique_players
          }
          logger.debug(f'Main Extraction: Step: {step} Time for steps: {current_time} unit_counts: {unit_counts}')
        
        if obs.player_result or (FLAGS.max_steps != -1 and step >= FLAGS.max_steps):
          break

    except KeyboardInterrupt:
      return None

    except Exception as e:
      # some error during the two player extraction has occurred
      log_message = f'\n\nError in replay extraction.\nReplay: {str(replay_file)}\n{str(e)}'
      logger.error(log_message)
      return None

    finally: 
      kill_controller(controller)

    did_player_one_extraction_succeed = extract_just_player_info(extractor, replay_file, player_id=1)
    if not did_player_one_extraction_succeed:
        # some error occurred, so return without saving
        return None

    did_player_two_extraction_succeed = extract_just_player_info(extractor, replay_file, player_id=2)
    if not did_player_two_extraction_succeed:
        # some error occurred, so return without saving
        return None

  # No errors occurred during any of the extractions, so process and save the data
  extractor.process_extraction()
  extractor.save(FLAGS.data_save_dir, checkpoint_metadata=True)

  if FLAGS.profile:
    logger.info(stopwatch.sw)

  return None


def main(unused_argv):

  # setting up logger:
  log_dir = Path(FLAGS.data_save_dir)/'logs'
  if not log_dir.exists():
      log_dir.mkdir()
  # create file handler which logs all messages
  verbose_fh = logging.FileHandler(log_dir/f"verbose-extraction_{time.strftime('%Y_%m_%d_%H%M%S')}.log", mode='w+')
  verbose_fh.setLevel(logging.DEBUG)
  # create file handler which logs info and above messages
  file_logger = logging.FileHandler(log_dir/f"extraction_{time.strftime('%Y_%m_%d_%H%M%S')}.log", mode='w+')
  file_logger.setLevel(logging.INFO)
  # create console_logger handler with a debug level
  console_logger = logging.StreamHandler()
  console_logger.setLevel(logging.DEBUG)
  # create formatter and add it to the handlers
  file_logger.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M'))
  verbose_fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
  console_logger.setFormatter(logging.Formatter('%(levelname)s - %(asctime)s - %(message)s', datefmt='%m-%d %H:%M'))
  # add the handlers to the logger
  logger.addHandler(file_logger)
  logger.addHandler(verbose_fh)
  logger.addHandler(console_logger)

  if FLAGS.replay_dir and FLAGS.replay:
      raise RuntimeError('Either `replay` or `replay_dir` should be given but not both')

  if FLAGS.replay_dir:
    replay_file_list = get_replays_to_do()

  elif FLAGS.replay:
    replay_file_list = [FLAGS.replay]

  if FLAGS.max_replays >= 0 and len(replay_file_list) > FLAGS.max_replays:
    replay_file_list = replay_file_list[:FLAGS.max_replays]
  logger.debug(f'Processing {len(replay_file_list)} replay files (up to max_replays={FLAGS.max_replays})')
  logger.debug(f'Processing:\n{replay_file_list}')

  root_save_dir = Path(FLAGS.data_save_dir)
  if not root_save_dir.exists():
      # should we allow for making of parent directories too?
      # I'm going with no for now since this way we don't accidentally make a large dir tree if a wrong path is passed
      root_save_dir.mkdir()  

  # save a configuration file for this dataset extraction
  run_config = make_run_config(FLAGS, replay_file_list)
  with open(root_save_dir/'run_config.json', 'w+') as file:
      json.dump(run_config, file, indent=2, sort_keys=True)

  def get_output(command_str, command_label='Command'):
    try:
      out = subprocess.check_output(command_str.split(' ')).decode('ascii').strip()
    except subprocess.CalledProcessError as e:
      logger.warning(f'Could not extract {command_label} for metadata')
      out = '{command_label}_unknown'
    except Exception as e:
      log_message = f'\n\nWarning: An exception occurred when trying to {command_str}'
      logger.error(log_message)
    return out

  # extract the replays
  # From: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
  def chunks(file_list, n):
      for i in range(0, len(file_list), n):
          yield file_list[i: min(i+n, len(file_list) + 1)]

  for cur_file_list in chunks(replay_file_list, 300):
      joblib.Parallel(n_jobs=FLAGS.num_parallel_jobs, backend='multiprocessing')(
                        joblib.delayed(run_extract_app)(replay_file) for replay_file in replay_file_list)
      # Simple hack to hopefully kill zombie SC2 jobs
      # Kill all SC jobs
      get_output('pkill --full SC2_x64', 'Killing Zombie Process')

  # pull together the checkpointed metadata
  metadata_list = []
  for metadata_checkpoint_filename in (root_save_dir/'temp_metadata_dir').glob('*_metadata.json'):
      with open(metadata_checkpoint_filename, 'r') as checkpoint_file:
          metadata_list.extend(json.load(checkpoint_file))
  # Save the concatenated metadata into one
  with open(root_save_dir/'metadata.json', 'w+') as fh:
    json.dump(metadata_list, fh, indent=2, sort_keys=True)
  # Remove the metadata checkpoint files and dir 
  # NOTE: we are not removing the temp dir here, just in case
#   for metadata_checkpoint_filename in (root_save_dir/'temp_metadata_dir').glob('*_metadata.json'):
#      # removes the files
#      metadata_checkpoint_filename.unlink()
#   (root_save_dir/'temp_metadata_dir').rmdir()
  
  return 0  # Return successful

def run_extract_app(replay_file):
  try:
    # Add replay_file as last argument
    args_to_main = FLAGS(sys.argv) # Discard args to main as unused
    metadata = extract_replay(replay_file)
  except Exception as e:
    # some error during the player extraction has occurred
    log_message = f'\n\nError in replay extraction.\nReplay: {str(replay_file)}\n{str(e)}'
    logger.error(log_message)
    metadata = {}
  return metadata


def entry_point():  # Needed so setup.py scripts work.
  app.run(main)


if __name__ == "__main__":
  app.run(main)
