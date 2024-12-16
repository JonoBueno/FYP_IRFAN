import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Define the flux calculation function with ergs-to-meV conversion
def Flux(L, r):
    return L / (4 * np.pi * r**2)


# Load the main data file
d0 = pd.read_csv("TTDW5.txt", sep="\s+")

# Extract labels, masses, and distances
labels = d0["Name"]
Ms = d0["M"].values
rs = d0["D"].values

# Create the main plot and a single twin y-axis
fig, ax = plt.subplots(1, 2)

# Loop through each star's data
for label, r, M in zip(labels, rs, Ms):
    if M > 0.1:
        df = pd.read_csv(f"{M}.data", sep="\s+")
        ls = (
            (10 ** df["log_L"].values) * 3.828e33 * 6.242e8
        )  # Convert luminosity from ergs to meV
        age = df["star_age"].values
        F = [
            Flux(l, r * 3.086e18) for l in ls
        ]  # Calculate flux for each luminosity and distance

        # Plot flux values on the first y-axis
        ax[0].plot(age, F, label=f"Flux - {label}")
        ax[0].set_xscale("log")
        ax[0].set_xlabel(r"Star Age (Year)")
        ax[0].set_ylabel(r"Flux (meVcm-2s-1)", color="b")
        ax[0].tick_params(axis="y", labelcolor="b")

# Create a single second y-axis for luminosity

for label, r, M in zip(labels, rs, Ms):
    if M > 0.1:
        df = pd.read_csv(f"{M}.data", sep="\s+")
        ll0 = df["log_L"].values
        age = df["star_age"].values

        # Plot luminosity values on the second y-axis
        ax[1].plot(age, ll0, linestyle="--", alpha=0.7)

ax[0].set_ylim(0, 0.6e5)

# Customize the second y-axis
ax[1].set_ylabel(r"log(L/(L$_\odot$))", color="r")
ax[1].tick_params(axis="y", labelcolor="r")
ax[1].set_xscale("log")
ax[1].set_xlabel(r"Star Age (Year)")
# Add a single legend for both y-axes
fig.legend(loc="upper left", fontsize=5)

# Title and show plot
plt.title("Flux and Luminosity over Star Age for All Stars (meV)")
plt.show()
