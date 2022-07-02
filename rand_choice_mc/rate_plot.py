import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_table('rate.txt', delim_whitespace=True)

for t in range(1,5):
    plt.plot(df['rate'], df[str(t)], 'o-', mfc='none',
         label=f'choose {t} item(s)')

plt.legend()
plt.xlabel('Probability of correct')
plt.ylabel('Average score')
plt.show()
