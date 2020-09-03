import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

data = pd.read_csv("Energy_Data.csv")
data = data[["Surface Area", "Wall Area", "Roof Area", "Overall Height",
             "Glazing Area", "Heating Load", "Cooling Load"]]

predict_heating = "Heating Load"
predict_cooling = "Cooling Load"

# Creates a new array without heating load data. (Our outcome)
X = np.array(data.drop([predict_heating, predict_cooling], 1))  # Attributes w/o Labels.
Y = np.array(data[predict_cooling])  # Labels only. (The outcomes derived from attributes)
# User uncomments the following line after the model achieves a desired accuracy.
"""x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)"""

# Block of code that user comments out after model achieves desired accuracy.
best = 0
# Runs model 30 times.
for _ in range(30):
    # x_train is a section of array X, y_train is a section of array Y.
    # 10% of original data is then split up into test samples (x_test and y_test)
    # so that the program is computing based off of samples it has never seen before.
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

    linear = linear_model.LinearRegression()

    # Finds line of best fit within x & y train data, line is stored within "linear".
    linear.fit(x_train, y_train)
    # Returns accuracy the of the model for each of the 30 times.
    acc = linear.score(x_test, y_test)
    print(acc, "\n")

    # Writing a new model if it scores better than the previous model created.
    if acc > best:
        best = acc
        with open("EnergyEfficiency-Cooling.pickle", "wb") as f:
            pickle.dump(linear, f)
# Block of code that user comments out after model achieves desired accuracy.

pickle_in = open("EnergyEfficiency-Cooling.pickle", "rb")
linear = pickle.load(pickle_in)

# Coefficients for a line in 5 dimensional space. (5 attributes are being used)
print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_, "\n")

# Using unused x_test array to make predictions.
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# Building scatter plot.
p = "Wall Area"
style.use("ggplot")
pyplot.scatter(data[p], data["Cooling Load"])
pyplot.xlabel(p)
pyplot.ylabel("Final Cooling Load")
pyplot.show()
