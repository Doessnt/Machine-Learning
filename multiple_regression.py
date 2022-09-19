import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# linear regeression
    # multi
# logisctic reg.
# polinomial reg.
# decision tree reg.
# random forest reg.

# Import
data = pd.read_csv('dataset/multiple_regression-dataset.csv')

exp_age = data.loc[:, ['exp', 'age'] ].values
salaries = data['salaries'].values.reshape(-1, 1)



# Algorithm
import sklearn.linear_model as lm
reg = lm.LinearRegression()

# Data split
import  sklearn.model_selection as ms

x_train, x_test, y_train, y_test = ms.train_test_split( exp_age, salaries, test_size = 1/3, random_state= 0)

# Train
reg.fit(x_train, y_train)

# Predict
y_pred = reg.predict( x_test )


print('Experiences', x_test)
print('Estimated', y_pred)



# Score
import sklearn.metrics as mt

score = mt.r2_score(y_test, y_pred)
print('score : ', score)


# Graph
plt.scatter(exp_age[:,0], salaries, color='r')
plt.scatter(x_test[:, 0], y_pred, color='b')
plt.show()










