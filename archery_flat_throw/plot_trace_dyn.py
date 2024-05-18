import matplotlib.pyplot as plt 
import math
import typing as tp
import numpy as np 
from plot_trace import compute_init_speed, compute_trace_points
from arrow_dynamics import ArrowDynamics, ArrowModel

############ DEFINES ############
W = 40     # J  elastic energy
R = 0.95   # thermal efficiency
K = 5.0    # g  effective mass
H = 1.0    # m  starting height
EN_FACT = 1.5   # J/lb  estimated elastic energy divided by force at constant draw
RADIUS = 3.8  # mm
RESI_COEF = 1.0  # resistence coefficient
TIMESTEP = 0.0001  # s  integration time step

#################################

# physical constants
g_const = 9.8   # m/s^2


def compute_drop_dyn(W, R, K, m, D, radius, resi_coef, timestep):
    model = ArrowModel(m, radius, resi_coef)
    d = ArrowDynamics(model)
    v0 = compute_init_speed(W, R, K, m)
    d.velocity = np.array([v0, 0.0])
    coords = d.get_trace_for_distance(D, timestep, 1.0)
    if coords[-1][0] >= D:
        return -coords[-1][1]
    else:
        return np.nan

def plot_trace_m_sweep():
    masses = np.arange(18.0, 37.0, 6.0)
    fig = plt.figure(figsize=[9.6,4.8])
    for i, m in enumerate(masses):
        dyn = ArrowDynamics(ArrowModel(m, RADIUS, RESI_COEF))
        v0 = compute_init_speed(W, R, K, m)
        dyn.set_height(H)
        dyn.set_horizontal_speed(v0)

        points = dyn.get_trace_for_height(TIMESTEP)
        xs, ys = zip(*points)
        plt.plot(xs, ys, c=f'C{i}', label=f'$m$={m} g')

        # no-resistance case
        ts, xs, ys = compute_trace_points(H, v0)
        plt.plot(xs, ys, '--', c=f'C{i}')
        
    plt.xlabel('$x$ (m)')
    plt.ylabel('$y$ (m)')
    plt.legend()
    plt.show()


def plot_drop_m_F_sweep():
    masses = np.arange(10, 42, 0.5)
    forces = list(range(20, 45, 5))
    DISTANCE = 18  # m

    for f in forces:
        print('force', f)
        drops = [-compute_drop_dyn(f*EN_FACT, R, K, m, DISTANCE, RADIUS, RESI_COEF, TIMESTEP) for m in masses]
        plt.plot(masses, drops, label=f'$F={f}$ lb')
    
    for rate in np.arange(0.6, 1.01, 0.1):
        ms = [rate*f for f in forces]
        drops = [-compute_drop_dyn(f*EN_FACT, R, K, rate*f, DISTANCE, RADIUS, RESI_COEF, TIMESTEP) for f in forces]
        plt.plot(ms, drops, 'o:', mfc='none')
        plt.annotate(f'{rate:.1f} g/lb', [ms[-1], drops[-1]])
    
    plt.xlabel('$m$ (g)')
    plt.ylabel('drop (m)')
    plt.legend()
    plt.show()


def plot_drop_resi_m_sweep():
    masses = np.arange(18.0, 37, 6.0)
    coeffs = np.arange(0.0, 80, 1.0)
    DISTANE = 18
    for m in masses:
        print(f'{m=}')
        drops = [-compute_drop_dyn(W, R, K, m, 18, RADIUS, resi, TIMESTEP) for resi in coeffs]
        plt.plot(coeffs, drops, label=f'$m$={m} g')
    
    plt.xlabel('$C$ (dimensionless)')
    plt.ylabel('drop (m)')
    plt.legend()
    plt.show()


def plot_drop_resi_F_sweep():
    forces = np.arange(20., 46., 5.)
    coeffs = np.arange(0.0, 80, 1.0)
    DISTANE = 18
    for f in forces:
        print(f'{f=}')
        m = f * 0.8   # 0.8 g/lb
        drops = [-compute_drop_dyn(f * EN_FACT, R, K, m, 18, RADIUS, resi, TIMESTEP) for resi in coeffs]
        plt.plot(coeffs, drops, label=f'$F$={f} lb')
    
    plt.xlabel('$C$ (dimensionless)')
    plt.ylabel('drop (m)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    ...
    # plot_trace_m_sweep()
    # plot_drop_m_F_sweep()
    # plot_drop_resi_m_sweep()
    plot_drop_resi_F_sweep()