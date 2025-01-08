import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

labels = [
    0.5,
    0.6,
    0.7,
    0.8,
    0.8,
    0.9,
    1,
    1.1,
    1.2,
    ]

df = []
df1 = []
masses = []
pred = 1 # float(input("What is the mass ="))


for label in labels:

    if label != pred:
        masses += [label]
        df += [pd.read_csv(f"0p0/{label}.data", sep="\s+")]
    df1 += [pd.read_csv(f"0p0/{label}.data", sep="\s+")]
masses = np.array(masses)

ages = []
lums = []
ages1 = []
lums1 = []
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

for d in df1:
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

    ages1 += [d["star_age"]]
    lums1 += [d["log_Lneu"]]

min_length = min(len(lum) for lum in lums)
ages = np.log([age[:min_length] for age in ages])
lums = np.array([lum[:min_length] for lum in lums])

min_length = min(len(lum) for lum in lums1)
ages1 = np.log([age[:min_length] for age in ages1])
lums1 = np.array([lum[:min_length] for lum in lums1])

from sklearn.linear_model import LinearRegression as LnR

output = np.array(
    [
        *(i for z in zip(ages.T, lums.T) for i in z),
    ]
).T
print(output.shape)
model = LnR()
model.fit(np.array([masses]).T, output)

# n = -2
# prediction = model.predict([[masses[n]]])
prediction = model.predict([[pred]])
predicted_ages = prediction[:, ::2]
predicted_lums = prediction[:, 1::2]
print(prediction.shape)

plt.plot(predicted_ages[0], predicted_lums[0], label=f"predicted {pred}")

# plt.plot(ages[n], lums[n], label="real")
# plt.plot(ages[1], lums[1], label="grid")
# plt.plot(ages[2], lums[2], label="grid 0.4")
# plt.xscale("log")

for i, label in enumerate(labels):
    plt.plot(ages1[i], lums1[i], label=f"grid {label}")

# plt.plot(ages1[6], lums1[6], label=f"grid {2.9}")
plt.ylabel('Luminosity')
plt.xlabel('Star Age')
plt.legend(fontsize=6)

plt.show()