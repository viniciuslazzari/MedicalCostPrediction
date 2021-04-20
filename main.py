import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./data.csv')

sex = {'female': 0, 'male': 1}
smoker = {'no': 0, 'yes': 1}
region = {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}


def filterData(data):
    data['smoker'] = [smoker[item] for item in data['smoker']]
    data['sex'] = [sex[item] for item in data['sex']]
    data['region'] = [region[item] for item in data['region']]
    data['charges'] = [round(item, 2) for item in data['charges']]

    return data


data = filterData(data)

x = data[['age', 'sex', 'bmi', 'children', 'smoker']]
x.insert(0, 'coefficient', np.ones(len(data.index)))
y = data['charges']

theta = np.zeros(len(x.columns))

alpha = 0.00003
iters = 1000


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
        gradient = 0
        for j in range(len(theta)):
            for m in range(len(x)):
                gradient += loss[m] - x[m][j]
            theta[j] -= (alpha/len(x)) * gradient

        cost.append(getMSE(x, y, theta))

    return theta, cost


x = np.array(x)
y = np.array(y)
theta = np.array(theta).T

MSE = getMSE(x, y, theta)
theta, cost = gradientDescent(x, y, theta, alpha, iters)

print(theta)
plt.plot(list(range(iters)), cost, '-r')
plt.show()
