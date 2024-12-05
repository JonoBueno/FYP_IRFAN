import matplotlib.pyplot as plt
import pandas as pd
import random

def random_color():
    return "#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)])

# File names and labels
files = ['0.2.data', '0.3.data', '0.4.data', '0.5.data', '1.data', '2.data', '3.data', '5.data','8.data','10.data','20.data','30.data','60.data','150.data']
labels = ['0.2', '0.3', '0.4', '0.5', '1', '2', '3', '5','8','10','20','30','60','150']

# Loop through files and plot each one
for file, label in zip(files, labels):
    df = pd.read_csv(file, sep='\s+')
    plt.plot(df['star_age'], df['log_L'], color=random_color(), label=label)

# Set plot scale, limits, and labels
plt.xscale('log')
plt.xlim(1e3, 1e12)
plt.ylim(-5, 12.5)

plt.ylabel('Log(Ly/Ls)')
plt.xlabel('Star age (Year)')
plt.legend(title='Stellar Mass',fontsize='xx-small',loc='upper right')

# Display the plot
plt.show()
