import matplotlib.pyplot as plt
import pandas as pd

df1=pd.read_csv('n13.dat',sep='\s+')
df2=pd.read_csv('o15.dat',sep='\s+')
df3=pd.read_csv('f17.dat',sep='\s+')

print(df1,'\n',df2,'\n',df3)

plt.plot(df1['x'],df1["y"],color='r')
plt.plot(df2['x'],df2["y"],color='b')
plt.plot(df3['x'],df3["y"],color='green')

plt.title('Normalize energy spectrum as a function of energy')
plt.ylabel('Normalized energy spectrum')
plt.xlabel('Energy (MeV)')
plt.legend(['13N','15O','17F'],title='Source')

plt.show()