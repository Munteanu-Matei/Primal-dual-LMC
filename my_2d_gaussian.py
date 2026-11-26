"""
Sampling from a 2D truncated Gaussian
"""

import os
import sys
import time

import numpy as np
import torch



CWD = os.path.dirname(__file__)
sys.path.append(os.path.join(CWD, ".."))


ITERATIONS = int(5e6)
BURN_IN = int(4e6)

# Constraints
# f = jit(lambda x: jnp.inner(x - 2 * jnp.ones(2), x - 2 * jnp.ones(2)) / 0.5)
# g = jit(lambda x: jnp.array(relu(jnp.inner(x, x) - 1.0) - 0.001))

def ellipse(p, a, b):
    return p[..., 0] ** 2 / a**2 + p[..., 1] ** 2 / b**2

MIU = torch.tensor([2.0, 2.0], dtype=torch.float64)
f = lambda x: 0.5 * torch.sum((x - MIU) ** 2)
g = lambda x: torch.relu(ellipse(x, 0.5, 2.0) - 1.0) - 0.001

def U(x, lmbda):
    return f(x) + lmbda * g(x)

def rejection(N):
    samples = np.zeros((N, 2))
    n = 0

    while n < N:
        proposal = np.random.normal(loc=2.0, size=(100000, 2))
        dists = ellipse(proposal, 0.5, 2.0)
        accepted = proposal[dists <= 1.0][: N - n]
        samples[n : n + accepted.shape[0]] = accepted
        n += accepted.shape[0]

    return samples


np.random.seed(1234)
start = time.process_time()
rejection_samples = rejection(BURN_IN)
print(f"Rejection sampling finished processing in {time.process_time()-start} seconds.")


def pd_lmc(f, g, iterations, step_size_x, step_size_lmbda, initial_x, initial_lmbda):
    x = torch.tensor(initial_x, dtype=torch.float64, requires_grad=True)
    lmbda = torch.tensor(initial_lmbda, dtype=torch.float64)

    x_traj = np.zeros((iterations, 2))
    lmbda_traj = np.zeros(iterations)

    for i in range(iterations):
        loss_x = U(x, lmbda)

        loss_x.backward()

        lmbda.data = torch.clamp(lmbda.data + step_size_lmbda * g(x.data), min=0.0)

        x.data = x.data - step_size_x * x.grad.data + (2 * step_size_x) ** 0.5 * torch.randn_like(x)
        x.grad.zero_()

        x_traj[i] = x.data.numpy()
        lmbda_traj[i] = lmbda.item()

    return x_traj, lmbda_traj



torch.manual_seed(1234)
start = time.process_time()
pdlmc_x, pdlmc_lambda = pd_lmc(
    f=f,
    g=g,
    iterations=ITERATIONS,
    step_size_x=1e-4,
    step_size_lmbda=2e-1,
    initial_x=np.zeros(2),
    initial_lmbda=np.array(0.0),
)
print(f"PD-LMC finished processing in {time.process_time()-start} seconds.")

np.savez(
    os.path.join(CWD, "2d_gaussian"),
    pdlmc_x=pdlmc_x,
    pdlmc_lambda=pdlmc_lambda,
    rejection_x=rejection_samples,
)
