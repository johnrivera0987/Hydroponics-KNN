# Importing libraries
import pandas as pd
import numpy as np
import pickle
import time
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt

# Reading CSV datasets
acc_df = pd.read_csv("MOCK_DATA.csv")
acc_df.fillna(0, inplace=True)
acc_df['Acceptable'] = True
unacc_df = pd.read_csv("MOCK_DATA-1.csv")
unacc_df.fillna(0, inplace=True)
unacc_df['Acceptable'] = False
df = acc_df.append(unacc_df, ignore_index=True)
shuffle(df)

# Separating into features and labels
X = df.loc[:, df.columns != 'Acceptable']
y = df.loc[:, df.columns == 'Acceptable']

start1 = time.time()
bestAcc = 0
# Training the best model using pickle and for loop
for _ in range(1000):

    # Separating training and testing set
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=1)

    # Scaling features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.astype(np.float32))
    X_test = scaler.transform(X_test.astype(np.float32))

    # Training KNN model
    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(X_train, y_train.values.ravel())
    acc = model.score(X_test, y_test.values.ravel())
    print(str(_) + " Accuracy: " + str(acc))
    if acc > bestAcc:
        bestAcc = acc
        with open("bestModel.pickle", "wb") as f:
            pickle.dump(model, f)
stop1 = time.time()

# Loading best model
pickle_in = open("bestModel.pickle", "rb")
model = pickle.load(pickle_in)

# Classification report and model accuracy
y_pred = model.predict(X_test)
classification_report = classification_report(y_test, y_pred)
accuracy = model.score(X_test, y_test)
print()
print("CLASSIFICATION REPORT")
print(classification_report)
print(f"MODEL ACCURACY: {round(bestAcc * 100, 2)}%")
print()
print(f"TRAINING TIME: {round(stop1 - start1, 2)}s")
disp = plot_confusion_matrix(model, X_test, y_test)
print(disp)
plt.show()

# Getting user input
pH_level = float(input('Input pH Level: '))
ec_level = float(input('Input EC Level: '))
area_of_lettuce = float(input('Input Area of Lettuce: '))

# Preprocessing user input
inputs = [[pH_level, ec_level, area_of_lettuce]]
new_inputs = scaler.transform(inputs)
new_inputs = pd.DataFrame(new_inputs, columns=(df.columns[0:3]))

# Predicting from user input
start2 = time.time()
pred = model.predict(new_inputs)
stop2 = time.time()
print()
print(f"INPUTS: pH Level = {pH_level}, EC Level = {ec_level}, Area of Lettuce = {area_of_lettuce}")
print(f"PREDICTION: ACCEPTABLE = {pred}")
print(f"PREDICTION TIME: {start2 - stop2} s")
if pred:
    print()
    print("ACTION: CLOSE VALVE")
else:
    print()
    print("ACTION: OPEN VALVE")

# Getting new CSV file
filename = str(input("Input CSV File Name: "))
new_df = pd.read_csv(filename)
new_df.fillna(0, inplace=True)

# Predicting acceptable values from CSV file
arr1 = []
arr2 = []
for i in new_df.index:
    pH_level = new_df['pH_level'][i]
    ec_level = new_df['ec_level'][i]
    area_of_lettuce = new_df['area_of_lettuce'][i]
    temp_df = scaler.transform([[pH_level, ec_level, area_of_lettuce]])
    temp_df = pd.DataFrame(temp_df, columns=df.columns[0:3])
    pred = model.predict(temp_df)
    arr1.append(pred)
    if pred:
        arr2.append('CLOSE VALVE')
    else:
        arr2.append('OPEN VALVE')
arr1 = pd.DataFrame(arr1)
arr2 = pd.DataFrame(arr2)
new_df['Acceptable'] = arr1
new_df['Action'] = arr2
print(new_df)