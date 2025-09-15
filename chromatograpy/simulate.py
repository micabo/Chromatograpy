import numpy as np


def simulate_column_1(n_cells: int, n_steps: int, n_molecules: int, K: float):
    # Reformat K to a fractional entity
    frac_bound = K / (1 + K)

    # Initial state (zero everywhere)
    stationary_phase = np.zeros(n_cells)
    mobile_phase = np.zeros(n_cells + n_steps)

    # Loading of mobile phase in cell just before the stationary phase
    mobile_phase[n_cells] = n_molecules

    for step in range(1, n_steps + 1):
        # Equilibration
        for cell in range(n_cells):
            n_tot = stationary_phase[cell] + mobile_phase[cell + step]
            n_bound = np.floor(n_tot * frac_bound)
            n_free = n_tot - n_bound
            stationary_phase[cell] = n_bound
            mobile_phase[cell + step] = n_free

    return mobile_phase


def fraction_bound(K, bound=0, capacity=None):
    if capacity is None:
        capacity_term = 1
    else:
        capacity_term = np.exp(bound / capacity)
    return K / (capacity_term + K)
