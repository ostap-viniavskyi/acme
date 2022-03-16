"""NGU JAX actors."""
from typing import Optional

import dm_env
import jax

from acme import adders
from acme import types
from acme.agents.jax import actors
from acme.jax import networks as network_lib
from acme.jax import utils
from acme.jax import variable_utils
from .actor_core import NGUActorCore


class NGUActor(actors.GenericActor):
    """A generic actor implemented on top of ActorCore.

    An actor based on a policy which takes observations and outputs actions. It
    also adds experiences to replay and updates the actor weights from the policy
    on the learner.
    """

    def __init__(
            self,
            actor: NGUActorCore,
            random_key: network_lib.PRNGKey,
            variable_client: Optional[variable_utils.VariableClient],
            adder: Optional[adders.Adder] = None,
            jit: bool = True,
            backend: Optional[str] = 'cpu',
            per_episode_update: bool = False
    ):
        """Initializes a feed forward actor.

        Args:
          actor: actor core.
          random_key: Random key.
          variable_client: The variable client to get policy parameters from.
          adder: An adder to add experiences to.
          jit: Whether or not to jit the passed ActorCore's pure functions.
          backend: Which backend to use when jitting the policy.
          per_episode_update: if True, updates variable client params once at the
            beginning of each episode
        """
        super(NGUActor, self).__init__(actor, random_key, variable_client, adder, jit, backend, per_episode_update)
        self._observe = actor.observe

    def select_action(self,
                      observation: network_lib.Observation) -> types.NestedArray:
        action, self._state = self._policy(self._params[0], observation, self._state)
        return utils.to_numpy(action)

    def observe_first(self, timestep: dm_env.TimeStep):
        self._random_key, key = jax.random.split(self._random_key)
        self._state = self._init(key)
        if self._adder:
            self._adder.add_first(timestep)
        if self._variable_client and self._per_episode_update:
            self._variable_client.update_and_wait()

    def observe(self, action: network_lib.Action, next_timestep: dm_env.TimeStep):
        # TODO: fix parameter indexing by number
        intrinsic_reward, self._state = self._observe(self._params[1], action, next_timestep, self._state)
        if self._adder:
            extras = self._get_extras(self._state)
            extras['intrinsic_rewards'] = intrinsic_reward

            self._adder.add(
                action, next_timestep, extras=extras)

    def update(self, wait: bool = False):
        if self._variable_client and not self._per_episode_update:
            self._variable_client.update(wait)
