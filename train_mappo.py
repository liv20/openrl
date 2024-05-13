# train_ppo.py

import numpy as np

from openrl.envs.common import make
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent
from openrl.configs.config import create_config_parser

from openrl.utils.callbacks.eval_callback import EvalCallback
from openrl.utils.callbacks.callbacks import CallbackList


# from sustaingym.envs.building import MultiAgentBuildingEnv, ParameterGenerator

def train(env_name, shift, seed = 0):
    # Add code for reading configuration files.
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()

    # Create MPE environment using asynchronous environment where each agent runs independently.
    if env_name == "simple_spread":
        # Modify params in config namespace
        cfg.seed = seed

        if shift == "random_action":
            envs = dict()
            shifts = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
            for random_action_prob in shifts:
                envs[random_action_prob] = make(
                    env_name,
                    env_num=20,
                    asynchronous=True,
                    random_action_prob=float(random_action_prob),
                    )
            env = envs["0.0"]
        elif shift == "collision_penalty":
            envs = dict()
            shifts = ["1", "100"]
            for collision_penalty in shifts:
                envs[collision_penalty] = make(
                    env_name,
                    env_num=20,
                    asynchronous=True,
                    collision_penalty=int(collision_penalty)
                )
            env = envs["1"]
        else:
            raise ValueError(f"Invalid shift: {shift}")

    elif env_name == "building":
        # https://github.com/chrisyeh96/sustaingym/blob/main/examples/building/examples_multiagent.ipynb
        from sustaingym.envs.building import MultiAgentBuildingEnv, ParameterGenerator

        numofhours = 24 * (4)
        chicago = [20.4, 20.4, 20.4, 20.4, 21.5, 22.7, 22.9, 23, 23, 21.9, 20.7, 20.5]
        city = "chicago"
        filename = "building_data/Exercise2A-mytestTable.html"
        weatherfile = "building_data/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw"
        U_Wall = [2.811, 12.894, 0.408, 0.282, 1.533, 12.894, 1.493]
        params = ParameterGenerator(
            filename,
            weatherfile,
            city,
            U_Wall=U_Wall,
            ground_temp=chicago,
            shgc=0.568,
            ac_map=np.array([1, 1, 1, 1, 1, 0]),
            shgc_weight=0.1,
            ground_weight=0.7,
            full_occ=np.array([1, 2, 3, 4, 5, 0]),
            # reward_gamma=[0.1, 0.9],
            activity_sch=117.24,
        )  # Description of ParameterGenerator in bldg_utils.py
        env = MultiAgentBuildingEnv(params)
    else:
        raise ValueError(f"{env_name} not one of ['simple_spread', 'building']")
    # Distribution shift

    # Create neural network with hyperparameter configurations.
    net = Net(env, cfg=cfg, device="cuda")
    # Use wandb.
    agent = Agent(net, use_wandb=True, project_name=f"PPOAgent-{env_name}-{shift}")
    # Start training.
    agent.train(total_time_steps=5000000,
                callback=CallbackList([
                    EvalCallback(envs[s], eval_freq=10000, wandb_path=f"Eval/{s}") for s in
                        shifts
                ])
    )

    # Save trained agents.
    agent.save("./ppo_agent/")

if __name__ == "__main__":
    train("building", "???", seed = 0)
    # train("simple_spread", "collision_penalty", seed = 0)
