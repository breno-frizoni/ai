from linear_regression import LinearRegression
import numpy as np
import math, json
from matplotlib import pyplot as plt

with open('data/linear_regression_sample.json', 'r', encoding='utf-8') as f:
    dados = json.load(f)
x_train = np.array([]); y_train = np.array([]);
for d in dados['dados_caminhada']:
    x_train = np.append(x_train, [d['x']])
    y_train = np.append(y_train, [d['y']])
modelo = LinearRegression(x_train, y_train)
y_hat = np.array([modelo.model(x_train[i]) for i in range(x_train.shape[0])])
plt.scatter(x_train, y_train)
plt.plot(x_train, y_hat)
plt.show()
modelo.gradientDescentDataTrain(0.01, 1000)
y_hat = np.array([modelo.model(x_train[i]) for i in range(x_train.shape[0])])
plt.scatter(x_train, y_train)
plt.plot(x_train, y_hat)
plt.show()




