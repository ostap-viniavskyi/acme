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

"""Defines distributed and local NGU agents, using JAX."""

from typing import Callable, Optional

from acme import specs
from acme.utils import counting
from .builder import NGUBuilder
from .config import NGUConfig
from .local_layout import LocalLayout as NGULocalLayout
from .networks import make_behavior_policy, NGUNetworks

NetworkFactory = Callable[[specs.EnvironmentSpec], NGUNetworks]


class NGU(NGULocalLayout):
    """Local agent for NGU.

    This implements a single-process NGU agent. This is a simple Q-learning
    algorithm that generates data via a (epsilon-greedy) behavior policy, inserts
    trajectories into a replay buffer, and periodically updates its policy by
    sampling these transitions using prioritization.
    """

    def __init__(
            self,
            spec: specs.EnvironmentSpec,
            networks: NGUNetworks,
            config: NGUConfig,
            seed: int,
            workdir: Optional[str] = '~/acme',
            counter: Optional[counting.Counter] = None,
    ):
        ngu_builder = NGUBuilder(networks, config)
        super().__init__(
            seed=seed,
            environment_spec=spec,
            builder=ngu_builder,
            networks=networks,
            policy_network=make_behavior_policy(networks, config),
            workdir=workdir,
            min_replay_size=32 * config.sequence_period,
            samples_per_insert=1.,
            batch_size=config.batch_size,
            num_sgd_steps_per_step=config.sequence_period,
            counter=counter,
        )
