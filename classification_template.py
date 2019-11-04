# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

#Split data into training set and test set
from sklearn.model_selection import train_test_split
inputTrain, inputTest, outputTrain, outputTest = train_test_split(X, y, test_size=0.25, random_state=0)

#Feature scaling
#method for normalizing the range of features (X variables)
from sklearn.preprocessing import StandardScaler
inputScalar = StandardScaler()
inputTrain = inputScalar.fit_transform(inputTrain) #calculate mean and std_dev then scale
inputTest = inputScalar.transform(inputTest) # just scale

# Fit classifier to the training set
#Create classifier here

#Predicting test set results
prediction = classifier.predict(inputTest)

# Make the confusion matrix
# Used for indicated correct preedictions
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(outputTest, prediction)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = inputTrain, outputTrain
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

