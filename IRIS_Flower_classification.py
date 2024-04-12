import numpy as np
import pandas as pd
from sklearn.cluster import k_means
import matplotlib.pyplot as plt
import  seaborn as sns

iris = sns.load_dataset('iris')
iris['species'],categories=pd.factorize(iris['species'])
print(iris.describe())
# print(iris.head())
print(iris.isna().sum())
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(iris.petal_length,iris.petal_width,iris.species)
ax.set_xlabel('Petal_Length_Cm')
ax.set_ylabel('Petal_Width_Cm')
ax.set_zlabel('species')
plt.title('3d Scatter Plot Example')
plt.show()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(iris.sepal_length,iris.sepal_width,iris.species)
ax.set_xlabel('Sepal_Length_Cm')
ax.set_ylabel('Sepal_Width_Cm')
ax.set_zlabel('species')
plt.title('3d Scatter Plot Example')
plt.show()