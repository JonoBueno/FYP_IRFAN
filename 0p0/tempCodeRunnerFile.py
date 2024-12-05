import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model as lm

def F(L,r):
    return L/(4*np.pi*r**2)

d0=pd.read_csv('TTDW5.txt',sep='\s+')

print(d0)
# File names and labels
startHb = []
endHb = []
#labels = d0['Name'].values
Ms = d0['M'].values
rs = d0['D'].values

#print(labels)
#print(rs)
#for M in Ms:
#    print(f'{M}.data')

#plt.scatter(Ms,rs)
plt.show()
