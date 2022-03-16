from functools import partial

import haiku as hk
import jax
import jax.nn as nn
import jax.numpy as jnp
import numpy as np
import optax


class MLP(hk.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = hk.Linear(output_size=64)
        self.fc2 = hk.Linear(output_size=128)
        self.fc3 = hk.Linear(output_size=1)

    def __call__(self, x: jnp.array):
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        return self.fc3(x)


def model_forward(x: jnp.array):
    model = MLP()
    return model(x)


def data_gen():
    batch_size = 32
    n_features = 3
    noise = 0.0

    w = np.array([1.0, 2.0, 3.0])[:, None]

    while True:
        xs = np.random.randn(batch_size, n_features)
        ys = np.dot(xs, w) + noise * np.random.randn(batch_size, 1)
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
        return jnp.mean(jnp.sum((ys - ys_pred) ** 2, axis=-1))

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
    train()
