"""Use the optional JAX backend (`pip install extended-einsum[jax]`)."""

import jax.numpy as jnp

import extended_einsum as xe


def main() -> None:
    source = jnp.arange(6.0).reshape(2, 3)
    result = xe.exp(source).materialize()

    assert jnp.allclose(result, jnp.exp(source))
    print(result)


if __name__ == "__main__":
    main()
