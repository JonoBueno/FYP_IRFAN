import matplotlib.pyplot as plt
import pandas as pd

df1=pd.read_csv('cnosurvival.tab.txt',sep='\s+')
print(df1)

plt.plot(df1['E'],df1["pp"],color='r')
plt.plot(df1['E'],df1["B8"],color='b')
plt.plot(df1['E'],df1["N13"],color='purple')
plt.plot(df1['E'],df1["O15"],color='pink')

plt.title('CNO reactions and the survival probabilities \n as a function of energy')
plt.ylabel('Probability')
plt.xlabel('Energy (MeV)')
plt.legend(['pp','B8','N13','O15'],title='Source')

plt.show()