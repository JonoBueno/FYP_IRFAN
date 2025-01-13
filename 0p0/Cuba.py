import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

labels = [
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.8,
    0.9,
    1,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5,
    1.6,
    1.7,
    1.8,
    1.9,
    2,
    2.1,
    2.2,
    2.3,
    2.4,
    2.5,
    2.6,
    2.7,
    2.8,
    2.9,
    3,
]
df = []
masses = []

for label in labels:
    masses += [label]
    df += [pd.read_csv(f"0p0/{label}.data", sep="\s+")]
masses = np.array(masses)

ages = []
lums = []

for d in df:
    condition = 10 * d["log_LH"] > 1 * 10 * d["log_L"]
    if condition.any():
        start_h_burning = d.loc[condition, "star_age"].iloc[0]
    else:
        start_h_burning = None

    condition = d["center_h1"] < 1e-6
    if condition.any():
        end_h_burning = d.loc[condition, "star_age"].iloc[0]
    else:
        end_h_burning = None

    d = d[(d["star_age"] > start_h_burning) & (d["star_age"] < end_h_burning)]

    ages += [d["star_age"]]
    lums += [d["log_Lneu"]]

min_length = min(len(lum) for lum in lums)
ages = np.log10([age[:min_length] for age in ages])
lums = np.array([lum[:min_length] for lum in lums])

# for a, l, label in zip(ages, lums, labels):
#     plt.plot(a, l, label=label)
# plt.legend(fontsize=6)
# plt.xscale('log')
# plt.show()

from mpl_toolkits import mplot3d

# fig = plt.figure(figsize = (10,10))
# ax = plt.axes(projection='3d')

fig = plt.figure(figsize = (8,8))
ax = plt.axes(projection='3d')
ax.grid()


for i,label in  enumerate(labels):
    t = label
    x = lums[i]
    y = ages[i]

    ax.plot3D(x, y, t)
    ax.set_title('3D Parametric Plot')

    # Set axes label
    ax.set_xlabel(r'luminosity $log(L_\nu /L_\odot)$', labelpad=20)
    ax.set_ylabel(r'Star Age [$10^x$yr]', labelpad=20)
    ax.set_zlabel(r'Mass $M_\odot$', labelpad=20)

plt.show()