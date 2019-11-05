# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

path = './Skin_NonSkin.txt'
if not os.path.exists(path):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt'
    urllib.request.urlretrieve(url, path)

# Importing the dataset
dataset = pd.read_csv(path, sep='\t', names=['B', 'G', 'R', 'skin'])
X = dataset.iloc[:, [0, 2]].values
y = dataset.iloc[:, 3].values

# check missing values
dataset.isna().sum()

# Split data into training set and test set
from sklearn.model_selection import train_test_split

inputTrain, inputTest, outputTrain, outputTest = train_test_split(X, y, test_size=.2, random_state=1)

# Feature scaling
# method for normalizing the range of features (X variables)
from sklearn.preprocessing import StandardScaler

inputScalar = StandardScaler()
inputTrain = inputScalar.fit_transform(inputTrain)  # calculate mean and std_dev then scale
inputTest = inputScalar.transform(inputTest)  # just scale

# Fit classifier to the training set
# Create classifier here
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=1)
classifier.fit(inputTrain, outputTrain)

# Predicting test set results
prediction = classifier.predict(inputTest)

# Make the confusion matrix
# Used for indicated correct preedictions
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(outputTest, prediction)

# Visualising the Training set results
from matplotlib.colors import ListedColormap

X_set, y_set = inputTrain, outputTrain
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

print(inputTrain)
print(outputTrain)

# 5 Fold Split
# First merge xtrain and ytrain so that we can easily divide into 5 chunks
data = np.concatenate([inputTrain,outputTrain],axis = 1)

# Divide our data to 5 chunks
chunks = np.split(data, 5)

datadict = {'fold1': {'train': {'x': None, 'y': None}, 'val': {'x': None, 'y': None}, 'test': {'x': xtest, 'y': ytest}},
            'fold2': {'train': {'x': None, 'y': None}, 'val': {'x': None, 'y': None}, 'test': {'x': xtest, 'y': ytest}},
            'fold3': {'train': {'x': None, 'y': None}, 'val': {'x': None, 'y': None}, 'test': {'x': xtest, 'y': ytest}},
            'fold4': {'train': {'x': None, 'y': None}, 'val': {'x': None, 'y': None}, 'test': {'x': xtest, 'y': ytest}},
            'fold5': {'train': {'x': None, 'y': None}, 'val': {'x': None, 'y': None},
                      'test': {'x': xtest, 'y': ytest}}, }

for i in range(5):
    datadict['fold' + str(i + 1)]['val']['x'] = chunks[i][:, 0:3]
    datadict['fold' + str(i + 1)]['val']['y'] = chunks[i][:, 3:4]

    idx = list(set(range(5)) - set([i]))
    X = np.concatenate(itemgetter(*idx)(chunks), 0)
    datadict['fold' + str(i + 1)]['train']['x'] = X[:, 0:3]
    datadict['fold' + str(i + 1)]['train']['y'] = X[:, 3:4]


def writepickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def readpickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


writepickle(datadict, 'data.pkl')

print(datadict)

