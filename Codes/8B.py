import matplotlib.pyplot as plt
import pandas as pd

df3=pd.read_csv('b8spectrum.txt',sep='\s+',skiprows=15)
print(df3)

plt.title('Neutrino energy and best-estimated spectrum')
plt.xlabel('Neutrino energy (MeV)')
plt.ylabel('Best-estimated spectrum')
plt.plot(df3['E'],df3["p"])

plt.show()