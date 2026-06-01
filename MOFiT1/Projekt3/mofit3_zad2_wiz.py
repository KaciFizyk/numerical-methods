import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("psi_output.csv")

x = np.sort(df['x'].unique())
y = np.sort(df['y'].unique())

psi = df.pivot(index='y', columns='x', values='psi').values

plt.figure(figsize=(10, 5))
X, Y = np.meshgrid(x, y)
contour = plt.contour(X, Y, psi, levels=500, cmap='viridis')
plt.colorbar(contour, label='psi')
plt.title('Rozkład funkcji prądu (ψ)')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
