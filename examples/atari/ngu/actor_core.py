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

"""NGU actor."""
import dataclasses
from typing import Callable
from typing import Generic, Mapping

import chex
import dm_env
import jax
import jax.numpy as jnp
from rlax import episodic_memory_intrinsic_rewards
from rlax._src.exploration import IntrinsicRewardState

from acme.agents.jax import actor_core as actor_core_lib
from acme.jax import networks as networks_lib
from acme.jax import utils
from .config import NGUConfig
from .networks import EpsilonRecurrentPolicy


@chex.dataclass(frozen=True, mappable_dataclass=False)
class NGUActorState(Generic[actor_core_lib.RecurrentState]):
    rng: networks_lib.PRNGKey
    epsilon: jnp.ndarray
    beta: jnp.ndarray
    recurrent_state: actor_core_lib.RecurrentState
    intrinsic_reward_state: IntrinsicRewardState


@dataclasses.dataclass
class NGUActorCore(actor_core_lib.ActorCore):
    observe: Callable


def get_actor_core(
        recurrent_policy: EpsilonRecurrentPolicy[actor_core_lib.RecurrentState],
        observation_embed_network,
        initial_core_state: actor_core_lib.RecurrentState, config: NGUConfig
) -> NGUActorCore:
    """Returns ActorCore for NGU."""

    def select_action(params: networks_lib.Params,
                      observation: networks_lib.Observation,
                      state: NGUActorState[actor_core_lib.RecurrentState]):
        # TODO(b/161332815): Make JAX Actor work with batched or unbatched inputs.
        rng, policy_rng = jax.random.split(state.rng)
        observation = utils.add_batch_dim(observation)
        recurrent_state = utils.add_batch_dim(state.recurrent_state)
        action, new_recurrent_state = utils.squeeze_batch_dim(recurrent_policy(
            params, policy_rng, observation, recurrent_state, state.epsilon))
        return action, NGUActorState(rng, state.epsilon, new_recurrent_state, state.intrinsic_reward_state)

    initial_core_state = utils.squeeze_batch_dim(initial_core_state)

    def observe(params: networks_lib.Params,
                action: networks_lib.Action,
                next_timestep: dm_env.TimeStep,
                state: NGUActorState[actor_core_lib.RecurrentState]):
        rng, embedding_rng = jax.random.split(state.rng)

        observation = utils.add_batch_dim(next_timestep.observation)
        embedding = observation_embed_network.apply(params, rng, observation)

        intrinsic_reward, intrinsic_reward_state = episodic_memory_intrinsic_rewards(
            embedding,
            num_neighbors=config.episodic_memory_num_neighbors,
            reward_scale=float(state.beta),
            intrinsic_reward_state=state.intrinsic_reward_state,
            constant=config.episodic_memory_pseudo_counts,
            epsilon=config.episodic_memory_epsilon,
            cluster_distance=config.episodic_memory_cluster_distance,
            max_similarity=config.episodic_memory_max_similarity,
            max_memory_size=config.episodic_memory_max_size
        )

        if next_timestep.last():
            # reset intrinsic reward_state
            state = NGUActorState(rng, state.epsilon, state.recurrent_state, None)
        else:
            state = NGUActorState(rng, state.epsilon, state.recurrent_state, intrinsic_reward_state)
        return utils.squeeze_batch_dim(intrinsic_reward), state

    def init(
            rng: networks_lib.PRNGKey
    ) -> NGUActorState[actor_core_lib.RecurrentState]:
        rng, epsilon_rng = jax.random.split(rng)
        epsilon = jax.random.choice(epsilon_rng,
                                    jnp.logspace(1, 8, config.num_epsilons, base=0.4))
        # TODO: add beta selection mechanism
        beta = jnp.array(config.beta, dtype=jnp.float32)
        return NGUActorState(rng, epsilon, beta, initial_core_state, None)

    def get_extras(
            state: NGUActorState[actor_core_lib.RecurrentState]
    ) -> Mapping[str, jnp.ndarray]:
        # TODO: add true intrinsic reward
        return {'core_state': state.recurrent_state, 'intrinsic_rewards': jnp.array(0.)}

    return NGUActorCore(init=init, select_action=select_action,
                        observe=observe, get_extras=get_extras)
