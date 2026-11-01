"""Manual batched-QP experiment; run this file directly to execute it."""


def main():
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, vmap
    import qpax
    import timeit

    """
    solve batched non-negative least squares (nnls) problems

    min_x    |Fx - g|^2
    st        x >= 0
    """

    n = 22  # size of x
    m = 10  # rows in F

    # create data for N_qps random nnls problems
    N_qps = 10000
    Fs = jnp.array(np.random.randn(N_qps, m, n))
    gs = jnp.array(np.random.randn(N_qps, m))

    @jit
    def form_qp(F, g):
        # convert the least squares to qp form
        n = F.shape[1]
        Q = F.T @ F
        q = -F.T @ g
        G = -jnp.eye(n)
        h = jnp.zeros(n)
        A = jnp.zeros((0, n))
        b = jnp.zeros(0)
        return Q, q, A, b, G, h

    # create the QPs in a batched fashion
    Qs, qs, As, bs, Gs, hs = vmap(form_qp, in_axes=(0, 0))(Fs, gs)

    # create function for solving a batch of QPs
    batch_qp = jit(vmap(qpax.solve_qp_primal, in_axes=(0, 0, 0, 0, 0, 0)))

    xs = batch_qp(Qs, qs, As, bs, Gs, hs)


if __name__ == "__main__":
    main()
