import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define the flux calculation function with ergs-to-meV conversion
def Flux(L, r):
    return (L / (4 * np.pi * r**2))

labels = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

# Load the main data file
df1 = pd.read_csv('0.4.data', sep='\s+')
df2 = pd.read_csv('0.5.data', sep='\s+')

Ln1 = df1['log_Lneu'].values.reshape(-1,1)
Age1 = df1['star_age'].values.reshape(-1,1)
Ln2 = df2['log_Lneu'].values.reshape(-1,1)
Age2 = df2['star_age'].values.reshape(-1,1)
startage = []
endage = []
print(labels)


for file in labels:
    df = pd.read_csv(f'{file}.data',sep='\s+')
    startage.append(df['star_age'].iloc[0])
    endage.append(df['star_age'].iloc[-1])
    
print(startage)    
print(endage)

import sklearn.kernel_ridge as kr

model1 = kr(kernel='rbf', alpha=1.0, gamma=0.1)  # Adjust alpha and gamma as needed
model2 = kr(kernel='rbf', alpha=1.0, gamma=0.1)
mode3 = kr(kernel='rbf', alpha=1.0, gamma=0.1)

model1.fit(Age2,Ln2)
model2.fit(labels,startage)