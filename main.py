import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./data.csv')

smoker = {'no': 0, 'yes': 1}


def filterData(data):
    data = pd.get_dummies(data, columns=["sex", "region"], prefix=["sex", "region"])

    data['smoker'] = [smoker[item] for item in data['smoker']]
    data['charges'] = [round(item, 2) for item in data['charges']]

    return data


def normalize(df):
    result = df.copy()

    for feature in df.columns:
        maxValue = df[feature].max()
        minValue = df[feature].min()
        result[feature] = (df[feature] - minValue) / (maxValue - minValue)

    return result


def getMSE(x, y, theta):
    predY = np.sum(x * theta, 1)
    meanSquare = np.power((predY - y), 2)
    MSE = np.sum(meanSquare) / (2 * len(x))

    return MSE


def gradientDescent(x, y, theta, alpha, iters):
    cost = []

    for iteration in range(iters):
        predY = np.sum(x * theta, 1)
        loss = predY - y
        for j in range(len(theta)):
            gradient = 0
            for m in range(len(x)):
                gradient += loss[m] * x[m][j]
            theta[j] -= (alpha/len(x)) * gradient

        print(getMSE(x, y, theta))
        cost.append(getMSE(x, y, theta))

    return theta, cost


data = filterData(data)
data = normalize(data)

x = data.loc[:, data.columns != 'charges']
x.insert(0, 'coefficient', np.ones(len(data.index)))
y = data['charges']
theta = np.zeros(len(x.columns))

alpha = 0.001
iters = 3000

x = np.array(x)
y = np.array(y)
theta = np.array(theta).T

theta, cost = gradientDescent(x, y, theta, alpha, iters)

plt.plot(list(range(iters)), cost, '-r')
plt.show()
