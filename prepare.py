import numpy as np
import pandas as pd
from sklearn import datasets

iris = datasets.load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target_names[iris.target]

df = df[['target', 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]

df.to_csv('data.csv')