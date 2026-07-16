"""Use the optional JAX backend (`pip install extended-einsum[jax]`)."""

import jax.numpy as jnp

import extended_einsum as xe


def main() -> None:
    source_jax = jnp.arange(6.0).reshape(2, 3)
    source = xe.array(source_jax)
    result = xe.exp(source).materialize()

    assert jnp.allclose(result.backend_array, jnp.exp(source_jax))
    print(result.backend_array)


if __name__ == "__main__":
    main()
