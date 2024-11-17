import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('predictions.csv')
df["sum"] = df["formal_consequence"] + df["material_consequence"] + df["logical_consequence"]
#print(df.head)
df["sum"].value_counts().sort_index().plot()
plt.xlabel("Total bigram frequency")
plt.ylabel("Number of books")
plt.show()