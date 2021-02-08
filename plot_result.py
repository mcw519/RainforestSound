import seaborn as sns
import sys
import pandas as pd
import matplotlib.pyplot as plt

result_csv = sys.argv[1]
df = pd.read_csv(result_csv)
del df["recording_id"]

plt.figure()
ax = sns.heatmap(df.values)
plt.ylabel("samples")
plt.xlabel("species")
plt.show()