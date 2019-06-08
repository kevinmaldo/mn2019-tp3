#!/usr/bin/env python
import sentiment
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    c = sentiment.LeastSquaresClassifier()
    points = 100
    x = np.linspace(1, 500, points)
    y = x ** 2 + np.random.normal(0, 5000, x.shape)
    c.fit(x, y)
    sns.scatterplot(x, y, color='r')
    y_pred = c.predict(x)
    sns.lineplot(x, y_pred, color='b')
    plt.show()
    print("done")

if __name__ == "__main__":
    main()
