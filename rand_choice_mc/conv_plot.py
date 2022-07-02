import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_table('converge.txt', delim_whitespace=True)

for t in range(1,5):
    plt.plot(df['cycles'], df[str(t)], 'o-', mfc='none',
         label=f'choose {t} item(s)')

plt.legend()
plt.xlabel('Simulation cycles')
plt.ylabel('Average score')
plt.show()
