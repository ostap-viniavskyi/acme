import haiku as hk
import jax.numpy as jnp


class EmbeddingNetwork(hk.Module):
    """
    Fully connected embedding network for extraction of controllable state from observations.
    """

    def __init__(self, embedding_size: int, num_actions: int):
        super(EmbeddingNetwork, self).__init__(name='embedding_network')
        self._embedding_torso = hk.nets.MLP([16, 32, embedding_size])  # TODO: move constants to config
        self._pred_head = hk.Linear(num_actions)

    def embed(self, observation: jnp.array) -> jnp.array:
        """
        Embed a single observation
        Args:
            observation: jnp.array representing a single observation of environment

        Returns:
            embedding vector for an observation
        """
        return self._embedding_torso(observation)

    def __call__(self, observation_tm1: jnp.array, observation_t: jnp.array) -> jnp.array:
        """
        Embed two consecutive observations x_{t_1} and x{t} and predict action a_{t-1}
        Args:
            observation_tm1: observation x_{t-1}
            observation_t:  observation x_{t}

        Returns:
            prediction logits for discrete action a_{t_1}
        """
        emb_tm1 = self.embed(observation_tm1)
        emb_t = self.embed(observation_t)
        return self._pred_head(jnp.concatenate([emb_tm1, emb_t], axis=-1))


if __name__ == '__main__':
    import jax

    EMBEDDING_SIZE = 4
    INPUT_SIZE = 8
    BATCH_SIZE = 2
    NUM_ACTIONS = 6


    def embed_fn(observation: jnp.array) -> jnp.array:
        embed_net = EmbeddingNetwork(EMBEDDING_SIZE, NUM_ACTIONS)
        return embed_net.embed(observation)


    def action_pred_fn(observation_tm1: jnp.array, observation_t: jnp.array) -> jnp.array:
        embed_net = EmbeddingNetwork(EMBEDDING_SIZE, NUM_ACTIONS)
        return embed_net(observation_tm1, observation_t)


    rng = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(rng, 2)

    embed_fn_init, embed_fn_apply = hk.without_apply_rng(hk.transform(embed_fn))
    action_pred_init, action_pred_apply = hk.without_apply_rng(hk.transform(action_pred_fn))

    embed_params = embed_fn_init(key1, jnp.ones((BATCH_SIZE, INPUT_SIZE)))
    observations = jnp.ones((BATCH_SIZE, INPUT_SIZE))
    embeddings = embed_fn_apply(embed_params, observations)
    print(embeddings.shape)

    action_pred_params = action_pred_init(key2, jnp.ones((BATCH_SIZE, INPUT_SIZE)), jnp.ones((BATCH_SIZE, INPUT_SIZE)))
    observations_tm1, observations_t = jnp.ones((BATCH_SIZE, INPUT_SIZE)), jnp.ones((BATCH_SIZE, INPUT_SIZE))
    action_logits = action_pred_apply(action_pred_params, observations_tm1, observations_t)
    print(action_logits.shape)

    embeddings_tm1 = embed_fn_apply(action_pred_params, observations_tm1)
    print(embeddings_tm1)
