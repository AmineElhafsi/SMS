import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import time

import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import cm

def nelder_mead(
    objective_function,
    x_starts,
    initial_simplexes=None,
    scale_factors=None,
    tol=1e-6,
    max_iter=30,
    alpha=1.0,
    gamma=2.0,
    rho=0.5,
    sigma=0.5,
    bounds=None,
):
    """
    Nelder-Mead optimization algorithm with optional bounds.
    
    Parameters:
        objective_function: Function to minimize (assumed to support parallel evaluations).
        x_starts: Initial guesses (numpy arrays of shape (n,)).
        scale_factors: Scale factors for each dimension to initialize simplex.
        tol: Tolerance for stopping criterion.
        max_iter: Maximum number of iterations.
        alpha: Reflection coefficient.
        gamma: Expansion coefficient.
        rho: Contraction coefficient.
        sigma: Shrink coefficient.
        bounds: Optional bounds for each dimension as a list of tuples [(min, max), ...].
    
    Returns:
        x_best: The best solution found.
        f_best: The function value at the best solution.
    """

    num_runs = len(x_starts)
    n = len(x_starts[0])

    # initialize multiple simplexes
    if scale_factors is None:
        scale_factors = np.ones(n)

    if initial_simplexes is None:
        simplexes = np.stack([np.vstack([x + scale_factors[i]*np.eye(n)[i] for i in range(n)] + [x]) for x in x_starts])
    else:
        simplexes = initial_simplexes
        if simplexes.shape[1] != n+1:
            raise ValueError("Initial simplexes must have shape (num_runs, n+1, n)")
    f_values = objective_function(simplexes.reshape(-1, n)).reshape((num_runs, n+1))

    plt.figure()

    trajectories = []
    f_trajectories = []

    for iter in range(max_iter):
        print(f"Nelder-Mead Iteration {iter}")
        a = time.time()
        order = np.argsort(f_values, axis=1)
        simplexes = np.take_along_axis(simplexes, order[:, :, None], axis=1)
        f_values = np.take_along_axis(f_values, order, axis=1)

        #### Plot simplex ###
        norm = plt.Normalize(f_values.min(), f_values.max())
        cmap = cm.get_cmap('coolwarm')
        colors = cmap(norm(f_values))

        trajectories.append(simplexes)
        f_trajectories.append(f_values)
        # print("Simplex: ", simplexes)
        #####################

        centroids = np.mean(simplexes[:, :-1], axis=1)

        # evaluate conditions
        xr = centroids + alpha * (centroids - simplexes[:, -1, :])
        if bounds is not None:
            xr = np.clip(xr, bounds[:, 0], bounds[:, 1])
        fr = objective_function(xr)

        r_mask = (f_values[:, 0] <= fr) & (fr < f_values[:, -2]) # reflection
        e_mask = fr < f_values[:, 0] # expansion
        c_mask = ~r_mask & ~e_mask # contraction

        # apply procedures
        # reflection
        if np.any(r_mask):
            simplexes[r_mask, -1, :] = xr[r_mask]
            f_values[r_mask, -1] = fr[r_mask]

        # expansion
        if np.any(e_mask):
            xe = centroids + gamma * (xr - centroids)
            if bounds is not None:
                xe = np.clip(xe, bounds[:, 0], bounds[:, 1])
            fe = objective_function(xe)

            # case where expansion is better than reflection
            mask = e_mask & (fe < fr)
            simplexes[mask, -1, :] = xe[mask]
            f_values[mask, -1] = fe[mask]

            # case when reflection is better than expansion
            mask = e_mask & (fe >= fr)
            simplexes[mask, -1, :] = xr[mask]
            f_values[mask, -1] = fr[mask]

        if np.any(c_mask):
            xc = np.zeros((num_runs, n))

            # case when reflection is better than worst
            mask = c_mask & (fr < f_values[:, -1])
            xc[mask] = centroids[mask] + rho * (xr[mask] - centroids[mask])

            # case when reflection is worse than worst
            mask = c_mask & (fr >= f_values[:, -1])
            xc[mask] = centroids[mask] + rho * (simplexes[mask, -1, :] - centroids[mask])

            if bounds is not None:
                xc = np.clip(xc, bounds[:, 0], bounds[:, 1])
            fc = objective_function(xc)

            # contraction
            better_mask = fc < np.minimum(fr, f_values[:, -1])
            simplexes[better_mask, -1, :] = xc[better_mask]
            f_values[better_mask, -1] = fc[better_mask]

            # shrink
            shrink_mask = ~better_mask
            if np.any(shrink_mask):
                for i in range(1, n+1):
                    simplexes[shrink_mask, i, :] = simplexes[shrink_mask, 0, :] + sigma * (simplexes[shrink_mask, i, :] - simplexes[shrink_mask, 0, :])
                    if bounds is not None:
                        simplexes[shrink_mask, i, :] = np.clip(simplexes[shrink_mask, i, :], bounds[:, 0], bounds[:, 1])
                    f_values[shrink_mask, i] = objective_function(simplexes[shrink_mask, i])

        b = time.time()
        print(f"Nelder-Mead iteration took {b-a:.2f} seconds")
        print("Best value: ", f_values[:, 0])

    best_index = np.argmin(f_values[:, 0])

    trajectories.append(simplexes)
    f_trajectories.append(f_values)

    return simplexes[best_index, 0], f_values[best_index, 0], trajectories, f_trajectories

# def nelder_mead(#_nelder_mead_vectorized(
#     objective_function,
#     x_starts,
#     scale_factors=None,
#     tol=1e-6,
#     max_iter=30,
#     alpha=1.0,
#     gamma=2.0,
#     rho=0.5,
#     sigma=0.5,
# ):
#     """
#     Nelder-Mead optimization algorithm.
    
#     Parameters:
#         objective_function: Function to minimize (assumed to support parallel evaluations).
#         x_starts: Initial guesses (numpy arrays of shape (n,)).
#         scale_factors: Scale factors for each dimension to initialize simplex.
#         tol: Tolerance for stopping criterion.
#         max_iter: Maximum number of iterations.
#         alpha: Reflection coefficient.
#         gamma: Expansion coefficient.
#         rho: Contraction coefficient.
#         sigma: Shrink coefficient.
    
#     Returns:
#         x_best: The best solution found.
#         f_best: The function value at the best solution.
#     """

#     num_runs = len(x_starts)
#     n = len(x_starts[0])

#     # initialize multiple simplexes
#     if scale_factors is None:
#         scale_factors = np.ones(n)

#     simplexes = np.stack([np.vstack([x + scale_factors[i]*np.eye(n)[i] for i in range(n)] + [x]) for x in x_starts])
#     f_values = objective_function(simplexes.reshape(-1, n)).reshape((num_runs, n+1))

#     plt.figure()

#     trajectories = []
#     f_trajectories = []

#     for _ in range(max_iter):
#         a = time.time()
#         order = np.argsort(f_values, axis=1)
#         simplexes = np.take_along_axis(simplexes, order[:, :, None], axis=1)
#         f_values = np.take_along_axis(f_values, order, axis=1)
#         # simplex, f_values = simplex[order], f_values[order]

#         #### Plot simplex ###
#         # Create a colormap
#         norm = plt.Normalize(f_values.min(), f_values.max())
#         cmap = cm.get_cmap('coolwarm')
#         colors = cmap(norm(f_values))

#         # Get colors for the points
#         # plt.clf()
#         # for i in range(num_runs):
#         #     plt.triplot(simplexes[i, :, 0], simplexes[i, :, 1], 'ko-')
#         #     plt.scatter(simplexes[i, :, 0], simplexes[i, :, 1], color=colors[i], s=100)
#         #     plt.xlim(0, 1.2)
#         #     # plt.ylim(-0.1, 0.15)
#         #     plt.xlabel("v (m/s)")
#         #     plt.ylabel("theta (deg)")
#         #     plt.pause(0.1)
#         trajectories.append(simplexes)
#         f_trajectories.append(f_values)
#         print("Simplex: ", simplexes)
#         #####################

#         centroids = np.mean(simplexes[:, :-1], axis=1)
        

#         # evaluate conditions
#         xr = centroids + alpha * (centroids - simplexes[:, -1, :])
#         fr = objective_function(xr)

#         r_mask = (f_values[:, 0] <= fr) & (fr < f_values[:, -2]) # reflection
#         e_mask = fr < f_values[:, 0] # expansion
#         c_mask = ~r_mask & ~e_mask # contraction

#         # apply procedures
#         # reflection
#         if np.any(r_mask):
#             simplexes[r_mask, -1, :] = xr[r_mask]
#             f_values[r_mask, -1] = fr[r_mask]

#         # expansion
#         if np.any(e_mask):
#             xe = centroids + gamma * (xr - centroids)
#             fe = objective_function(xe)

#             # case where expansion is better than reflection
#             mask = e_mask & (fe < fr)
#             simplexes[mask, -1, :] = xe[mask]
#             f_values[mask, -1] = fe[mask]

#             # case when reflection is better than expansion
#             mask = e_mask & (fe >= fr)
#             simplexes[mask, -1, :] = xr[mask]
#             f_values[mask, -1] = fr[mask]

#         if np.any(c_mask):
#             xc = np.zeros((num_runs, n))

#             # case when reflection is better than worst
#             mask = c_mask & (fr < f_values[:, -1])
#             xc[mask] = centroids[mask] + rho * (xr[mask] - centroids[mask])

#             # case when reflection is worse than worst
#             mask = c_mask & (fr >= f_values[:, -1])
#             xc[mask] = centroids[mask] + rho * (simplexes[mask, -1, :] - centroids[mask])

#             fc = objective_function(xc)

#             # contraction
#             better_mask = fc < np.minimum(fr, f_values[:, -1])
#             simplexes[better_mask, -1, :] = xc[better_mask]
#             f_values[better_mask, -1] = fc[better_mask]

#             # shrink
#             shrink_mask = ~better_mask
#             if np.any(shrink_mask):
#                 for i in range(1, n+1):
#                     simplexes[shrink_mask, i, :] = simplexes[shrink_mask, 0, :] + sigma * (simplexes[shrink_mask, i, :] - simplexes[shrink_mask, 0, :])
#                     f_values[shrink_mask, i] = objective_function(simplexes[shrink_mask, i])

#         b = time.time()
#         print(f"Nelder-Mead iteration took {b-a:.2f} seconds")
#         print("Best value: ", f_values[:, 0])
    
        

#     # plt.show()
#     best_index = np.argmin(f_values[:, 0])

#     #
#     trajectories.append(simplexes)
#     f_trajectories.append(f_values)
#     #

#     return simplexes[best_index, 0], f_values[best_index, 0], trajectories, f_trajectories

def _nelder_mead(
    objective_function,
    x_start,
    scale_factors=None,
    tol=1e-6,
    max_iter=30,
    alpha=1.0,
    gamma=2.0,
    rho=0.5,
    sigma=0.5,
):
    """
    Nelder-Mead optimization algorithm.
    
    Parameters:
        objective_function: Function to minimize (assumed to support parallel evaluations).
        x_starts: Initial guesses (numpy arrays of shape (n,)).
        scale_factors: Scale factors for each dimension to initialize simplex.
        tol: Tolerance for stopping criterion.
        max_iter: Maximum number of iterations.
        alpha: Reflection coefficient.
        gamma: Expansion coefficient.
        rho: Contraction coefficient.
        sigma: Shrink coefficient.
    
    Returns:
        x_best: The best solution found.
        f_best: The function value at the best solution.
    """

    n = len(x_start[0])

    # initialize multiple simplexes
    if scale_factors is None:
        scale_factors = np.ones(n)

    simplex = np.vstack([x_start + scale_factors[i]*np.eye(n)[i] for i in range(n)] + [x_start])
    f_values = objective_function(simplex)

    for iteration in range(max_iter):
        order = np.argsort(f_values)
        simplex, f_values = simplex[order], f_values[order]

        centroid = np.mean(simplex[:-1], axis=0)
        
        # reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective_function(xr)

        if f_values[0] <= fr < f_values[-2]:
            simplex[-1], f_values[-1] = xr, fr
        elif fr < f_values[0]: # expansion
            xe = centroid + gamma * (xr - centroid)
            fe = objective_function(xe)
            if fe < fr:
                simplex[-1], f_values[-1] = xe, fe
            else:
                simplex[-1], f_values[-1] = xr, fr
        else: # contraction
            if fr < f_values[-1]:
                xc = centroid + rho * (xr - centroid)
            else:
                xc = centroid + rho * (simplex[-1] - centroid)

            fc = objective_function(xc)
            if fc < min(fr, f_values[-1]):
                simplex[-1], f_values[-1] = xc, fc
            else: # shrink
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    f_values[i] = objective_function(simplex[i])

        if np.std(f_values) < tol:
            break

                
