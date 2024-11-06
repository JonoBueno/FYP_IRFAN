import matplotlib.pyplot as plt
import pandas as pd

df3=pd.read_csv('intp1301.data',sep='\s+',skiprows=1)
df3=df3.loc[1:44:1]
print(df3)

plt.title('Neutrino energy and best-estimated spectrum')
plt.xlabel('Neutrino energy (MeV)')
plt.ylabel('Best-estimated spectrum')
plt.plot(df3['e1'],df3["nu"])

plt.show()
