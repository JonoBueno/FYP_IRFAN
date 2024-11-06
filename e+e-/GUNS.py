import numpy as np
import matplotlib.pyplot as plt

def fnu(E, T, m):
    return 1 / (np.exp(np.sqrt(E**2 - m**2) / T) + 1)

def dFdP(E, fnu, m):
    return ((E**2 - m**2) / (2 * np.pi**2)) * fnu

Tnu = 0.168
zs = [0, 0.07, 0.13, 0.27, 0.33, 0.47, 0.6, 0.8, 1.07, 1.34, 1.74, 2.41, 3.61]#, 12.83, 13.01, 13.13, 13.23, 13.3, 13.36, 13.41, 13.45, 13.48, 13.51, 13.53, 13.55, 13.57, 13.59, 13.6]
E1 = [np.arange(0, 2, 0.01), np.arange(8.6, 8.72, 0.001), np.arange(50, 50.02, 0.0001)]
ms = [0, 8.6, 50]

for m, E in zip(ms, E1):
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    for z in zs:
        Tz = Tnu * (1 + z)
        
        fn = fnu(E, Tz, m)
        dF = dFdP(E, fn, m) / 2.5555e-16

        ax[0].plot(E, fn, label=f"z = {z}")
        ax[1].plot(E, dF, label=f"z = {z}")
        print(f'Mass={m} z={z} Max dF/dE={max(dF):.4e}')


    ax[0].set_title(f"fnu vs E for m = {m}")
    ax[0].set_xlabel("E")
    ax[0].set_ylabel("fnu(E)")
    ax[0].legend(fontsize=5)

    ax[1].set_title(f"dF/dE vs E for m = {m}")
    ax[1].set_xlabel("E")
    ax[1].set_ylabel("dF/dE")
    ax[1].legend(fontsize=5, title='Redshift(z)')

    plt.show()