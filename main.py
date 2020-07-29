import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

weights = np.array([65.5, 66.3, 66.1, 65.9,
                    65.3, 65.6, 65.4, 66.0,
                    65.1, 65.0, 65.7, 65.8,
                    65.2, 66.5, 66.2])
days = np.array([-14, -13, -12, -11,
                 -10, -9, -8, -7,
                 -6, -5, -4, -3,
                 -2, -1, 0])
X = pd.DataFrame(days, columns=["days"])
target = pd.DataFrame(weights, columns=["weights"])
y = target["weights"]

lm = LinearRegression()
lm.fit(X, y)

predicted_days = [7, 14]
new_days = pd.DataFrame(np.array(predicted_days))

predicted_weights = lm.predict(new_days)
plt.scatter(days, weights)
regression_weights = lm.predict(X)
plt.plot(days, regression_weights, color="blue")
plt.plot(new_days, predicted_weights,
         color="red", marker="o", markersize=10)
for vtext_x, vtext_y in zip(predicted_days, predicted_weights):
    plt.text(vtext_x, vtext_y + 0.05, '%.2f KG' % vtext_y, ha='center', va='bottom', fontsize=11)

plt.xlabel("Days")
plt.ylabel("Weights(KG)")
plt.show()
