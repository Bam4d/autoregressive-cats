import argparse
import os
import sys

import ray
from griddly import gd
from griddly.util.rllib.callbacks import VideoCallbacks, WinLoseMetricCallbacks
from griddly.util.rllib.environment.core import RLlibEnv
from griddly.util.rllib.torch.agents.conv_agent import SimpleConvAgent
from griddly.util.rllib.torch.conditional_actions.conditional_action_policy_trainer import \
    ConditionalActionImpalaTrainer
from ray import tune
from ray.rllib.agents.callbacks import MultiCallbacks
from ray.rllib.models import ModelCatalog
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.registry import register_env

from autocats.clusters_generator import ClustersLevelGenerator

parser = argparse.ArgumentParser(description='Run experiments')

parser.add_argument('--debug', action='store_true', help='Debug mode')

parser.add_argument('--root-directory', default=os.path.expanduser("~/ray_results"),
                    help='root directory for all data associated with the run')
parser.add_argument('--num-gpus', default=1, type=int, help='Number of GPUs to make available to ray.')
parser.add_argument('--num-cpus', default=8, type=int, help='Number of CPUs to make available to ray.')

parser.add_argument('--num-workers', default=7, type=int, help='Number of workers')
parser.add_argument('--num-envs-per-worker', default=5, type=int, help='Number of workers')
parser.add_argument('--num-gpus-per-worker', default=0, type=float, help='Number of gpus per worker')
parser.add_argument('--num-cpus-per-worker', default=1, type=float, help='Number of gpus per worker')
parser.add_argument('--max-training-steps', default=20000000, type=int, help='Number of workers')
parser.add_argument('--train-batch-size', default=500, type=int, help='Training batch size')
parser.add_argument('--actions-per-step', default=1, type=int, help='Number of actions to produce per time-step')

parser.add_argument('--capture-video', action='store_true', help='enable video capture')
parser.add_argument('--video-directory', default='videos', help='directory of video')
parser.add_argument('--video-frequency', type=int, default=1000000, help='Frequency of videos')

parser.add_argument('--seed', type=int, default=69420, help='seed for experiments')

parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')

if __name__ == '__main__':

    args = parser.parse_args()

    sep = os.pathsep
    os.environ['PYTHONPATH'] = sep.join(sys.path)

    if args.debug:
        ray.init(include_dashboard=False, num_gpus=args.num_gpus, num_cpus=args.num_cpus, local_mode=True)
    else:
        ray.init(include_dashboard=False, num_gpus=args.num_gpus, num_cpus=args.num_cpus)

    env_name = "ray-griddly-env"

    register_env(env_name, RLlibEnv)
    ModelCatalog.register_custom_model("ImpalaCNN", SimpleConvAgent)

    wandbLoggerCallback = WandbLoggerCallback(
        project='autocats',
        api_key_file='~/.wandb_rc',
        dir=args.root_directory
    )

    actions_per_step = args.actions_per_step

    max_training_steps = args.max_training_steps
    gdy_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clusters-select-sep-cols.yaml')

    config = {
        'framework': 'torch',
        'seed': args.seed,
        'num_workers': args.num_workers,
        'num_envs_per_worker': args.num_envs_per_worker,
        'num_gpus_per_worker': float(args.num_gpus_per_worker),
        'num_cpus_per_worker': args.num_cpus_per_worker,

        'train_batch_size': args.train_batch_size,

        'callbacks': MultiCallbacks([
            VideoCallbacks,
            WinLoseMetricCallbacks,
        ]),

        'model': {
            'custom_model': 'ImpalaCNN',
            'custom_model_config': {
            }
        },
        'env': env_name,
        'env_config': {
            'generate_valid_action_trees': True,
            'level_generator': {
                'class': ClustersLevelGenerator,
                'config': {
                    'width': 6,
                    'height': 6,
                    'p_red': 0.7,
                    'p_green': 0.7,
                    'p_blue': 0.7,
                    'm_red': 4,
                    'm_blue': 4,
                    'm_green': 4,
                    'm_spike': 4
                }
            },
            'yaml_file': gdy_file,
            'global_observer_type': gd.ObserverType.SPRITE_2D,
            'max_steps': 1000,
            #'actions_per_step': actions_per_step
        },
        #'actions_per_step': actions_per_step,
        #'autoregression_mode': 'actions',
        'entropy_coeff_schedule': [
            [0, 0.01],
            [max_training_steps, 0.0]
        ],
        'lr_schedule': [
            [0, args.lr],
            [max_training_steps, 0.0]
        ],

    }

    if args.capture_video:
        real_video_frequency = int(args.video_frequency / (args.num_envs_per_worker * args.num_workers))
        config['env_config']['record_video_config'] = {
            'frequency': real_video_frequency,
            'directory': os.path.join(args.root_directory, args.video_directory)
        }

    stop = {
        "timesteps_total": max_training_steps,
    }

    result = tune.run(
        ConditionalActionImpalaTrainer,
        local_dir=args.root_directory,
        config=config,
        stop=stop,
        callbacks=[wandbLoggerCallback],
    )
