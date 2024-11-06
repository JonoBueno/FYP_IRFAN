import matplotlib.pyplot as plt
import pandas as pd

df1=pd.read_csv('7belineshape.dat',sep='\s+',skiprows=[0,1])
print(df1)

plt.plot(df1['E'],df1["p"],color='r')

df2=pd.read_csv('7BeExcited.txt',sep='\s+')
print(df2)

plt.plot(df2['E'],df2["p"],color='b')
plt.title('Probability distribution for the emitted neutrinos as a \n function of observed neutrino energy')
plt.ylabel('Probability')
plt.xlabel('Energy qobs-qlab (keV)')
plt.legend(['861.8 keV','384.3 keV'],title='Peak')

plt.show()

