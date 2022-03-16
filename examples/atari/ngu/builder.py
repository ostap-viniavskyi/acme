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

"""NGU Builder."""
from typing import Callable, Iterator, List, Optional

import jax
import optax
import reverb

import acme
from acme import adders
from acme import core
from acme import specs
from acme.adders import reverb as adders_reverb
from acme.agents.jax import builders
from acme.datasets import reverb as datasets
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
from .actor import NGUActor
from .actor_core import get_actor_core
from .config import NGUConfig
from .learning import NGULearner
from .networks import NGUNetworks


class NGUBuilder(builders.ActorLearnerBuilder):
    """NGU Builder.

    This is constructs all of the components for Recurrent Experience Replay in
    Distributed Reinforcement Learning (Kapturowski et al.)
    https://openreview.net/pdf?id=r1lyTjAqYX.
    """

    def __init__(self,
                 networks: NGUNetworks,
                 config: NGUConfig,
                 logger_fn: Callable[[], loggers.Logger] = lambda: None, ):
        """Creates NGU learner, a behavior policy and an eval actor.

        Args:
          networks: NGU networks, used to build core state spec.
          config: a config with NGU hps
          logger_fn: a logger factory for the learner
        """
        self._networks = networks
        self._config = config
        self._logger_fn = logger_fn

        # Sequence length for dataset iterator.
        self._sequence_length = (
                self._config.burn_in_length + self._config.trace_length + 1)

        # Construct the core state spec.
        dummy_key = jax.random.PRNGKey(0)
        initial_state_params = networks.initial_state.init(dummy_key, 1)
        initial_state = networks.initial_state.apply(initial_state_params,
                                                     dummy_key, 1)
        core_state_spec = utils.squeeze_batch_dim(initial_state)
        self._extra_spec = {
            'core_state': core_state_spec
        }

    def make_learner(
            self,
            random_key: networks_lib.PRNGKey,
            networks: NGUNetworks,
            dataset: Iterator[reverb.ReplaySample],
            replay_client: Optional[reverb.Client] = None,
            counter: Optional[counting.Counter] = None,
    ) -> core.Learner:
        # The learner updates the parameters (and initializes them).
        return NGULearner(
            unroll=networks.unroll,
            initial_state=networks.initial_state,
            action_pred=networks.embedding_net_action_pred,
            batch_size=self._config.batch_size,
            random_key=random_key,
            burn_in_length=self._config.burn_in_length,
            discount=self._config.discount,
            importance_sampling_exponent=(
                self._config.importance_sampling_exponent),
            max_priority_weight=self._config.max_priority_weight,
            target_update_period=self._config.target_update_period,
            iterator=dataset,
            optimizer=optax.adam(self._config.learning_rate),
            action_pred_optimizer=optax.adamw(self._config.action_pred_learning_rate,
                                              weight_decay=self._config.action_pred_weight_decay),
            action_pred_clipping_factor=self._config.action_pred_clipping_factor,
            bootstrap_n=self._config.bootstrap_n,
            tx_pair=self._config.tx_pair,
            clip_rewards=self._config.clip_rewards,
            replay_client=replay_client,
            counter=counter,
            logger=self._logger_fn())

    def make_replay_tables(
            self,
            environment_spec: specs.EnvironmentSpec,
    ) -> List[reverb.Table]:
        """Create tables to insert data into."""
        # '''
        if self._config.samples_per_insert:
            samples_per_insert_tolerance = (
                    self._config.samples_per_insert_tolerance_rate *
                    self._config.samples_per_insert)
            error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
            limiter = reverb.rate_limiters.SampleToInsertRatio(
                min_size_to_sample=self._config.min_replay_size,
                samples_per_insert=self._config.samples_per_insert,
                error_buffer=error_buffer)
        else:
            limiter = reverb.rate_limiters.MinSize(self._config.min_replay_size)

        # add intrinsic rewards to extra_specs
        self._extra_spec['intrinsic_rewards'] = specs.Array(
            shape=environment_spec.rewards.shape,
            dtype=environment_spec.rewards.dtype,
            name='intrinsic_rewards'
        )

        return [
            reverb.Table(
                name=self._config.replay_table_name,
                sampler=reverb.selectors.Prioritized(
                    self._config.priority_exponent),
                remover=reverb.selectors.Fifo(),
                max_size=self._config.max_replay_size,
                rate_limiter=limiter,
                signature=adders_reverb.SequenceAdder.signature(
                    environment_spec, self._extra_spec))
        ]

    def make_dataset_iterator(
            self, replay_client: reverb.Client) -> Iterator[reverb.ReplaySample]:
        """Create a dataset iterator to use for learning/updating the agent."""
        dataset = datasets.make_reverb_dataset(
            table=self._config.replay_table_name,
            server_address=replay_client.server_address,
            batch_size=self._config.batch_size,
            prefetch_size=self._config.prefetch_size,
            num_parallel_calls=self._config.num_parallel_calls)
        return dataset.as_numpy_iterator()

    def make_adder(self,
                   replay_client: reverb.Client) -> Optional[adders.Adder]:
        """Create an adder which records data generated by the actor/environment."""
        return adders_reverb.SequenceAdder(
            client=replay_client,
            period=self._config.sequence_period,
            sequence_length=self._sequence_length,
            delta_encoded=True)

    def make_actor(
            self,
            random_key: networks_lib.PRNGKey,
            policy_network,
            observation_embed_network,
            adder: Optional[adders.Adder] = None,
            variable_source: Optional[core.VariableSource] = None) -> acme.Actor:

        # Create variable client.
        variable_client = variable_utils.VariableClient(
            variable_source,
            key='actor_variables',
            update_period=self._config.variable_update_period)

        # TODO(b/186613827) move this to
        # - the actor __init__ function - this is a good place if it is specific
        #   for NGU.
        # - the EnvironmentLoop - this is a good place if it potentially applies
        #   for all actors.
        #
        # Make sure not to use a random policy after checkpoint restoration by
        # assigning variables before running the environment loop.
        variable_client.update_and_wait()

        initial_state_key1, initial_state_key2, random_key = jax.random.split(
            random_key, 3)
        actor_initial_state_params = self._networks.initial_state.init(
            initial_state_key1, 1)
        actor_initial_state = self._networks.initial_state.apply(
            actor_initial_state_params, initial_state_key2, 1)

        actor_core = get_actor_core(policy_network,
                                    observation_embed_network,
                                    actor_initial_state,
                                    self._config)
        return NGUActor(
            actor_core, random_key, variable_client, adder, backend='cpu')
