import matplotlib.pyplot as plt
from sklearn import linear_model as lm
import numpy as np

Tnu = 0.168
zs = [0, 0.07, 0.13, 0.27, 0.33, 0.47, 0.6, 0.8, 1.07, 1.34, 1.74, 2.41, 3.61, 12.83, 13.01, 13.13, 13.23, 13.3, 13.36, 13.41, 13.45, 13.48, 13.51, 13.53, 13.55, 13.57, 13.59, 13.6]
Tz = []
zg = np.arange(0, 30, 0.1).reshape(-1, 1)  # Reshape for fitting

# Convert zs and Tz to numpy arrays and reshape zs for fitting
zs = np.array(zs).reshape(-1, 1)
for z in zs:
    Tz.append(Tnu * (1 + z))

Tz = np.array(Tz).reshape(-1, 1)  # Reshape Tz for fitting

# Ridge regression
reg = lm.Ridge(alpha=0.5)
reg.fit(zs, Tz)
Tg = reg.predict(zg)

# Plotting
plt.plot(zg, Tg, label='Ridge Regression Prediction')
plt.plot(zs, Tz, 'o', label='Original Data')
plt.xlabel('z')
plt.ylabel('Tz')
plt.title('Plot of Tz vs z')
plt.legend()
plt.show()
