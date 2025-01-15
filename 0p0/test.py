import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LnR

x = np.arange(0, 50, 1)
y = []

for val in x:
    y.append(val**2)

plt.plot(x, y, label='Original Data')

# Reshaping x and y to be 2D arrays
x = x.reshape(-1, 1)
y = np.array(y).reshape(-1, 1)

model = LnR()
model.fit(x, y)

# Predicting using the model
predictions = model.predict(x).flatten()

plt.plot(x, predictions, label='Linear Fit', color='red')
plt.xlabel('x');plt.ylabel('y')
plt.title('Example of Linear Regression on Non-linear Data')
plt.legend()
plt.show()
