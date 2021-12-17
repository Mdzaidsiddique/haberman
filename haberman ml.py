import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

haberman = pd.read_csv('C:/Users/masoo/Downloads/haberman.csv', header = None, na_values= '?')
haberman.head()
haberman.columns = ['age','operation year','axillary nodes','Survival Status']
haberman.columns
haberman.info()
haberman.isnull().sum()
haberman.shape
haberman.boxplot()
plt.boxplot(haberman['axillary nodes'])
haberman['axillary nodes'].describe()

sns.violinplot(x = 'Survival Status', y = 'axillary nodes', data = haberman)
plt.legend()
plt.show()

sns.boxplot(x = 'Survival Status', y = 'axillary nodes', data = haberman)

sns.pairplot(haberman)

y = haberman['Survival Status']
x = haberman.iloc[:,0:-1]
x.columns
x.shape
y.shape

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.20, random_state = 123)

from sklearn.linear_model import LinearRegression
lrmodel = LinearRegression()

lrmodel.fit(X = x_train, y = y_train)
lrmodel.coef_
lrmodel.intercept_

lrmodel.score(x_test, y_test)
pred = lrmodel.predict(x_test)
sns.scatterplot(x = y_test, y = pred)
sns.lineplot(y_test, pred)

from sklearn.tree import DecisionTreeRegressor
dtrmodel = DecisionTreeRegressor()
dtrmodel.fit(X = x_train, y = y_train)
dtrmodel.score(x_test, y_test)
from sklearn import tree
tree.plot_tree(dtrmodel)

from sklearn.ensemble import RandomForestRegressor
rfmodel = RandomForestRegressor()
rfmodel.fit(x_train, y_train)
rfmodel.score(x_test, y_test)

from math import sqrt
sqrt(306/2)
from sklearn.neighbors import KNeighborsRegressor
knmodel = KNeighborsRegressor(13)
knmodel.fit(x_train, y_train)
knmodel.score(x_test, y_test)

#from the scoring result we can say that multy linear model is best model for this data which score is 0.05601569442107024
#now Deploying the model


################################
uber = pd.read_csv('C:/Users/masoo/Downloads/Uber Request Data.csv')
uber.head()
uber.columns
uber.info()
uber.isnull().sum()
uber.shape
uber.boxplot()
