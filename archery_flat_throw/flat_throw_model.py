import matplotlib.pyplot as plt 
import math
import typing as tp
import numpy as np 


############ DEFINES ############
W = 40     # J  elastic energy
R = 0.95   # thermal efficiency
K = 5.0    # g  effective mass
H = 1.0    # m  starting height
EN_FACT = 1.5   # J/lb  estimated elastic energy divided by force at constant draw
#################################

# physical constants
g_const = 9.8   # m/s^2
equation_str = r'$rW=\frac{1}{2} (m+K) v^2$'


def compute_init_speed(W, R, K, m)->float:
    """
    Compute the initial speed in m/s
    """
    v = math.sqrt(2 * R * W / (m + K) * 1e3)  # 1e3: g -> kg
    return v


def compute_trace_points(H, v)->np.ndarray:
    """
    Returns: t, x, y
    """
    tmax = math.sqrt(2 * H / g_const)   # s
    t_step = 0.001   # s
    ts = np.arange(0.0, tmax+t_step, t_step)

    xs = v * ts
    ys = -0.5 * g_const * ts**2 + H
    return ts, xs, ys


def compute_drop(W, R, K, m, D):
    v0 = compute_init_speed(W, R, K, m)
    t = D / v0
    return g_const * t**2 / 2


def fixed_arg_str(W=None, r=None, K=None, m=None):
    lst = []
    if W is not None:
        lst.append(rf'$W={W}$ J')
    if K is not None:
        lst.append(rf'$K={K}$ g')
    if r is not None:
        lst.append(rf'$r={r}$')
    if m is not None:
        lst.append(rf'$m={m}$ g')
    return '\n'.join(lst)


def plot_trace_m_sweep(mass_lst):

    plt.figure(figsize=[10.0, 4.8])

    for m in mass_lst:
        v0 = compute_init_speed(W, R, K, m)
        ts, xs, ys = compute_trace_points(H, v0)
        plt.plot(xs, ys, label=rf'$m={m}$ g')

    plt.xlabel('$x$ (m)')
    plt.ylabel('$y$ (m)')
    plt.axhline(0, c='k', ls=':')
    plt.legend()

    plt.figtext(0.2, 0.5, r'$rW=\frac{1}{2} (m+K) v^2$', fontsize=16)

    plt.figtext(0.2, 0.3, 
        f'$W={W}$ J\n'
        f'$K={K}$ g\n'
        f'$r={R}$'
    )

    plt.tight_layout()
    plt.show()


def plot_speed_m_K_sweep(mass_lst):
    for k in range(2, 11):
        vs = [compute_init_speed(W,R,k,m) for m in mass_lst]
        plt.plot(mass_lst, vs, label=rf'$K={k}$ g')

    plt.figtext(0.6, 0.6, fixed_arg_str(W=W, r=R))
    
    plt.xlabel(f'$m$ (g)')
    plt.ylabel(f'$v$ (m/s)')

    plt.legend()
    plt.show()


def plot_speed_W_m_sweep(mass_lst):
    ws = list(range(30, 90))
    for m in mass_lst:
        vs = [compute_init_speed(w, R, K, m) for w in ws]
        plt.plot(ws, vs, label=f'$m={m}$ g')

    plt.figtext(0.75, 0.2, fixed_arg_str(r=R, K=K))
    plt.legend()
    plt.xlabel(r'$W$ (J)')
    plt.ylabel(r'$v$ (m/s)')
    plt.show()


def plot_speed_m_F_sweep(mass_list):
    mass_list = np.arange(8, 49, 0.1)
    forces = list(range(20, 45, 5))
    for f in forces:
        vs = [compute_init_speed(f * EN_FACT, R, K, m) for m in mass_list]
        plt.plot(mass_list, vs, label=rf'$F={f}$ lb')

    ratios = [0.6, 0.7, 0.8, 0.9, 1.0]
    for r in ratios:
        m_refs = [f * r for f in forces]
        v_refs = [compute_init_speed(f * EN_FACT, R, K, f * r) for f in forces]
        plt.plot(m_refs, v_refs, 'o:', mfc='none')
        plt.annotate(f'{r:.1f} g/lb', [m_refs[-1], v_refs[-1]], [m_refs[-1]+1, v_refs[-1]])
    
    plt.xlabel(r'$m$ (g)')
    plt.ylabel(r'$v_0$ (m/s)')

    plt.xlim(9, 46)
    plt.ylim(46, 66)
    plt.legend()
    plt.show()


def plot_Ek_m_F_sweep():
    mass_list = np.arange(8, 49, 0.1)
    forces = list(range(20, 45, 5))
    for f in forces:
        kins = [m/(m+K) * R * f * EN_FACT for m in mass_list]
        plt.plot(mass_list, kins, label=rf'$F={f}$ lb')

    for rate in np.arange(0.6, 1.01, 0.1):
        masses = [f * rate for f in forces]
        kins = [f*rate/(f*rate+K)*R*f*EN_FACT for f in forces]
        plt.plot(masses, kins, 'o:', mfc='none')
        plt.annotate(f'{rate:.1f} g/lb', [masses[-1], kins[-1]], [masses[-1], kins[-1]+2])

    plt.xlabel('$m$ (g)')
    plt.ylabel('$E_k$ (J)')
    plt.legend()
    plt.show()


def plot_v_F_R_sweep():
    mass_rate = 0.8
    forces = np.arange(20, 46, 0.1)
    rs = np.arange(0.98, 0.86, -0.02)
    for r in rs:
        vs = [compute_init_speed(f * EN_FACT, r, K, mass_rate * f) for f in forces]
        plt.plot(forces, vs, label=rf'$r={r:.2f}$')
    plt.xlabel('$F$ (lb)')
    plt.ylabel('$v_0$ (m/s)')
    plt.legend()
    plt.show()


def plot_Ekbow_m_F_sweep():
    masses = np.arange(10, 45, 0.1)
    forces = list(range(20, 45, 5))

    for f in forces:
        kins = [K/(m+K)*R*f*EN_FACT for m in masses]
        plt.plot(masses, kins, label=f'$F={f}$ lb')

    for r in np.arange(0.6, 1.01, 0.1):
        ms = [r*f for f in forces]
        kins = [K/(r*f+K)*R*f*EN_FACT for f in forces]
        plt.plot(ms, kins, 'o:', mfc='none')
        plt.annotate(f'{r:.1f} g/lb', [ms[-1]+1, kins[-1]])
    
    plt.xlabel('$m$ (g)')
    plt.ylabel('$E_k$ (bow) (J)')
    plt.legend()
    plt.show()


def plot_drop_m_F_sweep():
    masses = np.arange(10, 42, 0.1)
    forces = list(range(20, 45, 5))
    DISTANCE = 18  # m

    for f in forces:
        drops = [-compute_drop(f*EN_FACT, R, K, m, DISTANCE) for m in masses]
        plt.plot(masses, drops, label=f'$F={f}$ lb')
    
    for rate in np.arange(0.6, 1.01, 0.1):
        ms = [rate*f for f in forces]
        drops = [-compute_drop(f*EN_FACT, R, K, rate*f, DISTANCE) for f in forces]
        plt.plot(ms, drops, 'o:', mfc='none')
        plt.annotate(f'{rate:.1f} g/lb', [ms[-1], drops[-1]])
    
    plt.xlabel('$m$ (g)')
    plt.ylabel('drop (m)')
    plt.legend()
    plt.show()


def plot_drop_F_r_sweep():
    forces = np.arange(20, 45.1, 0.1)
    mass_rate = 0.8
    DISTANCE = 18
    rs = np.arange(0.98, 0.86, -0.02)
    for r in rs:
        drops = [-compute_drop(EN_FACT * f, r, K, mass_rate * f, DISTANCE) for f in forces]
        plt.plot(forces, drops, label=rf'$r={r:.2f}$')

    # plt.axhline(-compute_drop(EN_FACT * 30, 0.94, K, mass_rate * 30, DISTANCE))

    plt.xlabel('$F$ (lb)')
    plt.ylabel('drop (m)')
    plt.legend()
    plt.show()


def plot_v_F_K_sweep():
    forces = np.arange(20, 45.1, 0.1)
    mass_rate = 0.8
    Ks = np.arange(1.0, 9.1, 1.0)
    for k in Ks:
        vs = [compute_init_speed(f * EN_FACT, R, k, f * mass_rate) for f in forces]
        plt.plot(forces, vs, label=rf'$K={k}$ g')

    plt.axhline(compute_init_speed(30 * EN_FACT, R, 5.0, 30 * mass_rate))
    
    plt.xlabel('$F$ (lb)')
    plt.ylabel('$v_0$ (m/s)')
    plt.legend()
    plt.show()


def plot_drop_F_K_sweep():
    forces = np.arange(20, 45.1, 0.1)
    mass_rate = 0.8
    DISTANCE = 18
    Ks = np.arange(1.0, 9.1, 1.0)
    for k in Ks:
        drops = [-compute_drop(EN_FACT * f, R, k, mass_rate * f, DISTANCE) for f in forces]
        plt.plot(forces, drops, label=rf'$K={K}$ g')

    # plt.axhline(-compute_drop(EN_FACT * 30, 0.94, K, mass_rate * 30, DISTANCE))

    plt.xlabel('$F$ (lb)')
    plt.ylabel('drop (m)')
    plt.legend()
    plt.show()


def plot_Ek_F_K_sweep():
    mass_rate = 0.8
    forces = np.arange(20, 45.5, 0.1)
    Ks = np.arange(1., 9.1, 1.)
    for k in Ks:
        kins = [f*mass_rate/(f*mass_rate+k) * R * f * EN_FACT for f in forces]
        plt.plot(forces, kins, label=rf'$K={k}$ g')

    plt.xlabel('$F$ (lb)')
    plt.ylabel('$E_k$ (J)')
    plt.legend()
    plt.show()


def plot_Ekbow_F_K_sweep():
    mass_rate = 0.8
    forces = np.arange(20, 45.5, 0.1)
    Ks = np.arange(1., 9.1, 1.)
    for k in Ks:
        kins = [k/(f*mass_rate+k) * R * f * EN_FACT for f in forces]
        plt.plot(forces, kins, label=rf'$K={k}$ g')

    plt.xlabel('$F$ (lb)')
    plt.ylabel('$E_k$ (bow) (J)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    mass_lst = [18, 20, 22, 24, 26, 28, 30, 32, 34, 36]
    # plot_trace_m_sweep(mass_lst)
    # plot_speed_m_K_sweep(mass_lst)
    # plot_speed_W_m_sweep(mass_lst)
    # plot_speed_m_F_sweep(mass_lst)
    # plot_Ek_m_F_sweep()
    # plot_v_F_R_sweep()
    # plot_Ekbow_m_F_sweep()
    # plot_drop_m_F_sweep()
    # plot_drop_F_r_sweep()
    # plot_v_F_K_sweep()
    # plot_drop_F_K_sweep()
    # plot_Ek_F_K_sweep()
    plot_Ekbow_F_K_sweep()
