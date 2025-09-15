from typing import Optional

import numpy as np
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN


def plot_dist(index, kde, bounds, num_points=61):
    # generate grid of points within bounds
    grid = np.mgrid[
        tuple(slice(b[0], b[1], complex(num_points)) for b in bounds.T)
    ]
    grid_pts = grid.reshape(len(bounds), -1).T

    # Evaluate KDE on the grid
    density = kde(grid_pts.T)

    # plot density as pcolormesh
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.pcolormesh(*grid, density.reshape(num_points, num_points))
    plt.colorbar()
    plt.title(f"KDE for Iteration {index+1}")
    plt.savefig("debug_density/{}.png".format(index))

def cross_entropy_kde_multi(
    objective_function,
    dim,
    num_samples=1000,
    num_elite=100,
    max_iters=20,
    bounds=None,
    minimize=False,
):
    assert bounds.shape[0] == dim

    samples = np.random.uniform(bounds[0], bounds[1], size=(num_samples, dim))

    for iteration in range(max_iters):
        # evaluate objective function
        sample_scores = objective_function(samples)
        if minimize:
            sample_scores = -sample_scores

        # select elites
        elite_indices = np.argsort(sample_scores)[-num_elite:]
        elite_samples = samples[elite_indices]

        # fit KDE to elite samples
        kde = gaussian_kde(elite_samples.T, bw_method='silverman')

        # plot KDE
        plot_dist(iteration, kde, bounds)

        # Update samples by sampling from the KDE
        samples = kde.resample(num_samples).T
        samples = np.clip(samples, bounds[0], bounds[1])






