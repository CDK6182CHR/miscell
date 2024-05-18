import arrow_dynamics as dyn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from plot_trace import compute_init_speed

############ DEFINES ############
W = 40     # J  elastic energy
R = 0.95   # thermal efficiency
K = 5.0    # g  effective mass
H = 1.0    # m  starting height
EN_FACT = 1.5   # J/lb  estimated elastic energy divided by force at constant draw
RADIUS = 3.8  # mm
RESI_COEF = 0.0  # resistence coefficient
TIMESTEP = 0.001  # s  integration time step

#################################

# physical constants
g_const = 9.8   # m/s^2


def plot_dist_angle():
    MASS = 30  # g
    HEIGHT = 1.5  # m
    model = dyn.ArrowModel(30, RADIUS, RESI_COEF)
    energy = 40 * 1.5
    v0 = compute_init_speed(energy, R, K, MASS)
    print(f'init speed: {v0} m/s')

    angs = np.arange(-5.0, 8.0, 0.50)
    dists = []
    for ang in angs:
        rad = ang * np.pi / 180.0
        d = dyn.ArrowDynamics(model)
        d.velocity = np.array([v0 * np.cos(rad), v0 * np.sin(rad)])
        d.pos = np.array([0.0, HEIGHT])
        coords = d.get_trace_for_height(TIMESTEP)
        dists.append(coords[-1][0])

    plt.axhline(40.0, ls=':', c='k')
    plt.plot(angs, dists)
    plt.xlabel('Angle (degree)')
    plt.ylabel('Distance (m)')
    plt.show()


def plot_trace_angle():
    MASS = 30  # g
    HEIGHT = 1.5  # m
    model = dyn.ArrowModel(30, RADIUS, RESI_COEF)
    energy = 40 * 1.5
    v0 = compute_init_speed(energy, R, K, MASS)
    print(f'init speed: {v0} m/s')
    cmap = matplotlib.cm.get_cmap('cool')
    plt.figure(figsize=[9.6, 4.8])

    ang_range = (-1.0, 4.1)
    
    def normalize(x):
        """Normalize given x to [0,1] range"""
        return (x - ang_range[0]) / (ang_range[1] - ang_range[0])

    angs = np.arange(*ang_range, 0.20)
    for ang in angs:
        rad = ang * np.pi / 180.0
        d = dyn.ArrowDynamics(model)
        d.velocity = np.array([v0 * np.cos(rad), v0 * np.sin(rad)])
        d.pos = np.array([0.0, HEIGHT])
        coords = d.get_trace_for_height(TIMESTEP)
        xs,ys = zip(*coords)
        c = cmap(normalize(ang))
        plt.plot(xs, ys, c=c, label=f'{ang} deg')
        if np.abs(np.fmod(round(ang, 3), 1.0)) < 1e-5:
            print(ang)
            plt.annotate(f'{ang:.0f}', coords[-1], [coords[-1][0], -0.1], color='b',
                         arrowprops=dict(arrowstyle='-'))

    plt.axvline(40, ls=':', c='k')
    plt.axhline(0, ls=':', c='k')
    plt.xlabel('$x$ (m)')
    plt.ylabel('$z$ (m)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    ...
    # plot_dist_angle()
    plot_trace_angle()

