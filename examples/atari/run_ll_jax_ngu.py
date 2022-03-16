# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run JAX R2D2 on Atari."""

import inspect
import json
import logging
import os
from typing import Optional, Callable, Any, Mapping

import gym
import haiku as hk
import jax
import jax.numpy as jnp
from absl import app
from absl import flags

import acme
from acme import wrappers
from acme.jax import networks as networks_lib
from acme.utils import loggers
from acme.utils.loggers.tf_summary import TFSummaryLogger
from examples.atari import ngu

flags.DEFINE_string('level', 'LunarLander-v2', 'Which game to play.')
flags.DEFINE_integer('num_episodes', 10000, 'Number of episodes to train for.')
flags.DEFINE_string('exp_path', 'experiments/default', 'Run name.')

flags.DEFINE_integer('seed', 0, 'Random seed.')

FLAGS = flags.FLAGS


def make_ll_environment():
    env_name = "LunarLander-v2"

    env = gym.make(env_name)
    env = wrappers.GymWrapper(env)
    env = wrappers.SinglePrecisionWrapper(env)

    return env


def from_dict_to_dataclass(cls, data):
    return cls(
        **{
            key: (data[key] if val.default == val.empty else data.get(key, val.default))
            for key, val in inspect.signature(cls).parameters.items()
        }
    )


class RDQN(hk.RNNCore):
    """A simple recurrent network for testing."""

    def __init__(self, num_actions: int):
        super().__init__(name='my_network')
        self._torso = hk.Sequential([
            hk.Flatten(),
            hk.nets.MLP([50, 50]),
        ])
        self._core = hk.LSTM(20)
        self._head = networks_lib.PolicyValueHead(num_actions)

    def __call__(self, inputs, state):
        embeddings = self._torso(inputs)
        embeddings, new_state = self._core(embeddings, state)
        logits, _ = self._head(embeddings)
        return logits, new_state

    def initial_state(self, batch_size: int):
        return self._core.initial_state(batch_size)

    def unroll(self, inputs, state):
        embeddings = jax.vmap(self._torso)(inputs)  # [T D]
        embeddings, new_states = hk.static_unroll(self._core, embeddings, state)
        logits, _ = self._head(embeddings)
        return logits, new_states


def make_networks(spec, batch_size, embedding_size):
    """Creates networks used by the agent."""

    def forward_fn(inputs, state):
        model = RDQN(spec.actions.num_values)
        return model(inputs, state)

    def initial_state_fn(batch_size: Optional[int] = None):
        model = RDQN(spec.actions.num_values)
        return model.initial_state(batch_size)

    def unroll_fn(inputs, state):
        model = RDQN(spec.actions.num_values)
        return model.unroll(inputs, state)

    def embedding_net_action_pred_fn(observation_tm1: jnp.array, observation_t: jnp.array) -> jnp.array:
        embed_net = ngu.EmbeddingNetwork(embedding_size, spec.actions.num_values)
        return embed_net(observation_tm1, observation_t)

    def embedding_net_embed_fn(observation: jnp.array) -> jnp.array:
        embed_net = ngu.EmbeddingNetwork(embedding_size, spec.actions.num_values)
        return embed_net.embed(observation)

    return ngu.make_networks(
        forward_fn=forward_fn,
        initial_state_fn=initial_state_fn,
        unroll_fn=unroll_fn,
        embedding_net_action_pred_fn=embedding_net_action_pred_fn,
        embedding_net_embed_fn=embedding_net_embed_fn,
        env_spec=spec,
        batch_size=batch_size)


def make_tf_logger(
        workdir: str = '~/acme/',
        label: str = 'learner',
        save_data: bool = True,
        time_delta: float = 1.0,
        asynchronous: bool = False,
        print_fn: Optional[Callable[[str], None]] = print,
        serialize_fn: Optional[Callable[[Mapping[str, Any]], str]] = loggers.base.to_numpy,
        steps_key: str = 'steps',

) -> loggers.base.Logger:
    del steps_key
    if not print_fn:
        print_fn = logging.info

    terminal_logger = loggers.terminal.TerminalLogger(label=label, print_fn=print_fn)

    all_loggers = [terminal_logger]

    if save_data:
        all_loggers.append(loggers.csv.CSVLogger(directory_or_file=workdir, label=label))

    all_loggers.append(TFSummaryLogger(logdir=workdir, label=label))

    logger = loggers.aggregators.Dispatcher(all_loggers, serialize_fn)
    logger = loggers.filters.NoneFilter(logger)

    logger = loggers.filters.TimeFilter(logger, time_delta)
    return logger


def main(_):
    env = make_ll_environment()
    env_spec = acme.make_environment_spec(env)

    with open('config.json') as f:
        config_dict = json.load(f)
    if not os.path.exists(FLAGS.exp_path):
        os.makedirs(FLAGS.exp_path)
    with open(os.path.join(FLAGS.exp_path, 'config.json'), 'w') as f:
        json.dump(config_dict, f)

    config = from_dict_to_dataclass(ngu.NGUConfig, config_dict)

    agent = ngu.NGU(
        env_spec,
        networks=make_networks(env_spec, config.batch_size, config.observation_embed_dim),
        config=config,
        seed=FLAGS.seed)

    logger = make_tf_logger(FLAGS.exp_path)

    loop = acme.EnvironmentLoop(env, agent, logger=logger)
    loop.run(FLAGS.num_episodes)

    eval_agent = ngu.NGU(
        env_spec,
        networks=make_networks(env_spec, config.batch_size, config.observation_embed_dim),
        config=config,
        seed=FLAGS.seed)


if __name__ == '__main__':
    app.run(main)
