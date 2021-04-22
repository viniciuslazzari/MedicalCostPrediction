import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('./data.csv')

smoker = {'no': 0, 'yes': 1}


def filterData(df):
    df = pd.get_dummies(df, columns=['children', 'sex', 'region'], prefix=['children', 'sex', 'region'])

    df['smoker'] = [smoker[item] for item in df['smoker']]
    df['charges'] = [round(item, 2) for item in df['charges']]

    return df


def normalizeData(df):
    for feature in df.columns:
        maxValue = df[feature].max()
        minValue = df[feature].min()
        df[feature] = (df[feature] - minValue) / (maxValue - minValue)

    return df


def getMSE(x, y, theta):
    y_pred = np.sum(x * theta, 1)
    meanSquare = np.power((predY - y), 2)
    MSE = np.sum(meanSquare) / len(x)

    return MSE


def gradientDescent(x, y, theta, alpha, iters):
    cost = []

    for iteration in range(iters):
        y_pred = np.sum(x * theta, 1)
        loss = y_pred - y
        for j in range(len(theta)):
            gradient = 0
            for m in range(len(x)):
                gradient += loss[m] * x[m][j]
            theta[j] -= (alpha/len(x)) * gradient

        cost.append(getMSE(x, y, theta))

    return theta, cost


def testModel(x, y, theta):
    y_pred = np.sum(x * theta, 1)
    df = {'y_pred': y_pred, 'y': y}
    df = pd.DataFrame(df)

    return df


def getSumSquareError(x, y, theta):
    y_pred = np.sum(x * theta, 1)
    SSres = np.sum(np.power((y_pred - y), 2))

    return SSres


def getSumSquareTotal(y):
    SStot = np.sum(np.power(y - y.mean(), 2))

    return SStot


data = filterData(data)
data = normalizeData(data)

x = data.loc[:, data.columns != 'charges']
x.insert(0, 'coefficient', np.ones(len(data.index)))
y = data['charges']
theta = np.zeros(len(x.columns))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

alpha = 0.003
iters = 10000

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
theta = np.array(theta).T

theta, cost = gradientDescent(x_train, y_train, theta, alpha, iters)
result = testModel(x_test, y_test, theta)

plt.plot(list(range(iters)), cost, '-r')
plt.show()

R_square = 1 - (getSumSquareError(x_test, y_test, theta)/getSumSquareTotal(y_test))
print(R_square)
