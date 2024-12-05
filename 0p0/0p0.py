import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model as lm

def F(L,r):
    return L/(4*np.pi*r^2)

d0=pd.read_csv('TTDW5.txt',sep='\s+')

# File names and labels
files = ['0.2.data', '0.3.data', '0.4.data', '0.5.data', '1.data', '2.data', '3.data', '5.data','8.data','10.data','20.data','30.data','60.data','150.data']
labels = ['0.2', '0.3', '0.4', '0.5', '1', '2', '3', '5','8','10','20','30','60','150']
startHb= []
endHb= []

reg = lm.Ridge(alpha=.5)


for file in files:
    # Read the file into a DataFrame
    df = pd.read_csv(file, sep='\s+')

    # Find the start of hydrogen burning: values where the condition is true
    condition = 10 * df['log_LH'] > 1 * 10 * df['log_L']
    
    # Check if there are any values that meet the condition
    if condition.any():
        # Get the first value of 'star_age' where the condition is True
        start_h_burning = df.loc[condition, 'star_age'].iloc[0]
    else:
        start_h_burning = None  # Or assign a default value like None or -1 if no value is found
    
    # Append the result to the list
    startHb.append(start_h_burning)

    condition = df['center_h1']<1e-6

    if condition.any():
        # Get the first value of 'star_age' where the condition is True
        end_h_burning = df.loc[condition, 'star_age'].iloc[0]
    else:
        end_h_burning = None  # Or assign a default value like None or -1 if no value is found
    
    endHb.append(end_h_burning)

# Print the list of first 'star_age' values where hydrogen burning starts
print(startHb)
print(endHb)

for i,(file, label, Hb, eHb) in enumerate(zip(files, labels, startHb, endHb)):
    df = pd.read_csv(file, sep=r"\s+")
    data = np.array([df["star_age"], df["log_Lneu"]]).T
    data = data[data[:, 0] > Hb]
    data = data[data[:, 0] < eHb]
    plt.plot(
        data.T[0],
        data.T[1],
        color='red',
        label=label)
    
    data2 = np.array([df["star_age"], df["log_Lneu"]]).T
    data2 = data2[data2[:, 0] > eHb]
    plt.plot(
        data2.T[0],
        data2.T[1],
        color='b',
        )
    
    plt.text(Hb-1e2, df.loc[df['star_age'] == Hb, 'log_Lneu'].values[0] -1, label, color='blue', fontsize=12, ha='center')
    # Plot a dot at ZAMS for each star's data
    plt.plot(Hb, df.loc[df['star_age'] == Hb, 'log_Lneu'].values[0], 'ko') 
    # Add label for ZAMS only once on the last dataset
    if i == len(files) - 1:
        plt.text(Hb, df.loc[df['star_age'] == Hb, 'log_Lneu'].values[0] + 0.1, 'ZAMS', color='blue', fontsize=12, ha='center')
        
    # Plot a dot at TAMS for each star's data
    plt.plot(eHb, df.loc[df['star_age'] == eHb, 'log_Lneu'].values[0], 'ko') 
    # Add label for TAMS only once on the last dataset
    if i == len(files) - 1:
        plt.text(eHb, df.loc[df['star_age'] == eHb, 'log_Lneu'].values[0] + 2.1, 'TAMS', color='red', fontsize=12, ha='center')

    X = df[['star_age']].values  # 2D array for features (star_age as the single feature)
    y = df['log_Lneu'].values    # 1D array for target values


# Set plot scale, limits, and labels
plt.xscale('log')
plt.xlim(1e3, 1e12)
plt.ylim(-5, 12.5)
plt.ylabel('Log(Ln/Ls)')
plt.xlabel('Star age (Year)')

# Display the plot
plt.show()
