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

"""NGU Networks."""

import dataclasses
from typing import Any, Callable, Optional, Tuple

import haiku as hk
import rlax

from acme import specs
from acme import types
from acme.agents.jax import actor_core as actor_core_lib
from acme.jax import networks as networks_lib
from acme.jax import utils
from .config import NGUConfig

Epsilon = float
EpsilonRecurrentPolicy = Callable[[
                                      networks_lib.Params, networks_lib.PRNGKey, networks_lib
        .Observation, actor_core_lib.RecurrentState, Epsilon
                                  ], Tuple[networks_lib.Action, actor_core_lib.RecurrentState]]


@dataclasses.dataclass
class NGUNetworks:
    """Network and pure functions for the NGU agent.."""
    # Recurrent Universal Q-value approximator
    forward: networks_lib.FeedForwardNetwork
    unroll: networks_lib.FeedForwardNetwork
    initial_state: networks_lib.FeedForwardNetwork
    # embedding network
    embedding_net_action_pred: networks_lib.FeedForwardNetwork
    embedding_net_embed: networks_lib.FeedForwardNetwork


def make_networks(
        env_spec: specs.EnvironmentSpec,
        forward_fn: Any,
        initial_state_fn: Any,
        unroll_fn: Any,
        embedding_net_action_pred_fn: Any,
        embedding_net_embed_fn: Any,
        batch_size) -> NGUNetworks:
    """Builds functional NGU network from recurrent model definitions."""

    # Make networks purely functional.
    forward_hk = hk.transform(forward_fn)
    initial_state_hk = hk.transform(initial_state_fn)
    unroll_hk = hk.transform(unroll_fn)

    embedding_net_action_pred_hk = hk.transform(embedding_net_action_pred_fn)
    embedding_net_embed_hk = hk.transform(embedding_net_embed_fn)

    # Define networks init functions.
    def initial_state_init_fn(rng, batch_size):
        return initial_state_hk.init(rng, batch_size)

    dummy_obs_batch = utils.tile_nested(
        utils.zeros_like(env_spec.observations), batch_size)
    dummy_obs_sequence = utils.add_batch_dim(dummy_obs_batch)

    def unroll_init_fn(rng, initial_state):
        return unroll_hk.init(rng, dummy_obs_sequence, initial_state)

    def embedding_net_action_pred_init_fn(rng):
        return embedding_net_action_pred_hk.init(rng, dummy_obs_batch, dummy_obs_batch)

    # Make FeedForwardNetworks.
    forward = networks_lib.FeedForwardNetwork(
        init=forward_hk.init, apply=forward_hk.apply)
    unroll = networks_lib.FeedForwardNetwork(
        init=unroll_init_fn, apply=unroll_hk.apply)
    initial_state = networks_lib.FeedForwardNetwork(
        init=initial_state_init_fn, apply=initial_state_hk.apply)

    embedding_net_action_pred = networks_lib.FeedForwardNetwork(
        init=embedding_net_action_pred_init_fn, apply=embedding_net_action_pred_hk.apply)
    embedding_net_embed = networks_lib.FeedForwardNetwork(
        init=embedding_net_embed_hk.init, apply=embedding_net_embed_hk.apply)

    return NGUNetworks(
        forward=forward, unroll=unroll, initial_state=initial_state,
        embedding_net_action_pred=embedding_net_action_pred,
        embedding_net_embed=embedding_net_embed)


def make_atari_networks(batch_size, env_spec):
    """Builds default NGU networks for Atari games."""

    # TODO: fix this

    def forward_fn(x, s):
        model = networks_lib.R2D2AtariNetwork(env_spec.actions.num_values)
        return model(x, s)

    def initial_state_fn(batch_size: Optional[int] = None):
        model = networks_lib.R2D2AtariNetwork(env_spec.actions.num_values)
        return model.initial_state(batch_size)

    def unroll_fn(inputs, state):
        model = networks_lib.R2D2AtariNetwork(env_spec.actions.num_values)
        return model.unroll(inputs, state)

    return make_networks(env_spec=env_spec, forward_fn=forward_fn,
                         initial_state_fn=initial_state_fn, unroll_fn=unroll_fn,
                         batch_size=batch_size)


def make_behavior_policy(
        networks: NGUNetworks,
        config: NGUConfig,
        evaluation: bool = False) -> EpsilonRecurrentPolicy:
    """Selects action according to the policy."""

    def behavior_policy(params: networks_lib.Params, key: networks_lib.PRNGKey,
                        observation: types.NestedArray,
                        core_state: types.NestedArray,
                        epsilon):
        q_values, core_state = networks.forward.apply(
            params, key, observation, core_state)
        epsilon = config.evaluation_epsilon if evaluation else epsilon
        return rlax.epsilon_greedy(epsilon).sample(key, q_values), core_state

    return behavior_policy
