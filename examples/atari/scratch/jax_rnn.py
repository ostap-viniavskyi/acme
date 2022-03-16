import jax
import haiku as hk
import numpy as np
import jax.numpy as jnp
import optax
import jax.nn as nn
from functools import partial


class RNN(hk.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.gru_core = hk.GRU(64)
        self.fc_head = hk.Linear(1)

    def __call__(self, x: jnp.array, state):
        x, new_state = self.gru_core(x, state)
        return self.fc_head(x), new_state

    def initial_state(self, batch_size: int):
        return self.gru_core.initial_state(batch_size)

    def unroll(self, x, state):
        # x - [T, B, E]
        embeddings, new_states = hk.static_unroll(self.gru_core, x, state)
        embeddings = self.fc_head(embeddings)
        print(embeddings.shape, new_states.shape)
        return embeddings, new_states


def forward_fn(x: jnp.array, state):
    model = RNN()
    return model(x, state)


def initial_state_fn(batch_size):
    model = RNN()
    return model.initial_state(batch_size)


def unroll_fn(x, state):
    model = RNN()
    return model.unroll(x, state)


def data_gen():
    batch_size = 32
    seq_len = 64
    n_features = 3
    noise = 0.0
    ma_coeffs = np.array([])

    w = np.array([1.0, 2.0, 3.0])
    b = 1.

    while True:
        xs = np.random.randn(batch_size, n_features)
        ys = np.dot(xs, w) + noise * np.random.randn(batch_size)
        yield jnp.array(xs), jnp.array(ys)


def train():
    rng_key = jax.random.PRNGKey(42)
    lr = 0.001

    model = hk.without_apply_rng(hk.transform(model_forward))
    params = model.init(rng_key, x=jnp.ones((1, 3)))

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    def mse(params, xs, ys):
        ys_pred = model.apply(params, xs)
        return jnp.mean((ys - ys_pred) ** 2)

    @partial(jax.jit, static_argnames=['lr'])
    def update(params, opt_state, xs, ys, lr):
        loss, grads = jax.value_and_grad(mse)(params, xs, ys)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    losses = []
    for i, (xs, ys) in enumerate(data_gen(), start=1):
        ys_pred = model.apply(params, xs)
        params, opt_state, loss = update(params, opt_state, xs, ys, lr)
        losses.append(loss)

        if i % 100 == 0:
            print(np.mean(losses[-100:]))
            losses = []


if __name__ == '__main__':
    # xs, ys = next(data_gen())
    # print(xs, ys)
    # train()
    model_hk = hk.without_apply_rng(hk.transform(forward_fn))
    model_init_hk = hk.without_apply_rng(hk.transform(initial_state_fn))
    model_unroll_hk = hk.without_apply_rng(hk.transform(unroll_fn))

    rng_key = jax.random.PRNGKey(42)

    state_init_params = model_init_hk.init(rng_key, batch_size=2)
    state = model_init_hk.apply(state_init_params, batch_size=2)

    params = model_hk.init(rng_key, jnp.ones((2, 10)), state)
    y, state = model_hk.apply(params, jnp.ones((2, 10)), state)

    model_unroll_hk.apply(params, jnp.ones((3, 2, 10)), state)


    # state = model_init_hk.apply()