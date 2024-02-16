import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 


def plot_force_curve_excel(ks, zero_draw_inch):

    plt.figure(figsize=[10.8, 4.8])
    plt.subplot(121)

    xs = np.linspace(zero_draw_inch, 30, 1000)
    ys = np.polyval(ks, xs)

    plt.plot(xs, ys, label='3-fit')

    plt.xlabel('Draw (inch)')
    plt.ylabel('Force (lb)')

    plt.axvline(28, ls=':')

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.xlim(zero_draw_inch, xmax)
    plt.ylim(0, ymax)

    plt.fill_between(xs, ys, alpha=0.2)
    plt.text(17, 8, r'$W(x)=\int F(x) \mathrm{d} x$', fontsize=18)

    # plt.legend()

    # now, for integral

    plt.subplot(122)
    ksint = np.polyint(ks)
    ysint = np.polyval(ksint, xs)

    energy_zero = np.polyval(ksint, zero_draw_inch)
    ys_refined = ysint - energy_zero
    ys_joule = ys_refined * 0.0254 * 0.45359237 * 9.8

    plt.plot(xs, ys_joule)

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.xlim(zero_draw_inch, xmax)
    plt.ylim(0, ymax)

    plt.xlabel('Draw (inch)')
    plt.ylabel('Elastic energy (J)')
    plt.axvline(28, ls=':')

    plt.tight_layout()
    

ks = np.array([ 1.41949501e-05, -1.64290758e-03,  7.42532222e-02, -1.60211655e+00,
        1.73298486e+01, -6.43298022e+01])
zero_draw_inch = 6.93667543
plot_force_curve_excel(ks, zero_draw_inch)
plt.show()

