import pandas as pd 
import numpy as np
import csv 
import random
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

#creating csv files from the given excel 
#cols_fall = [2,3]

#data_xl_fall = pd.read_excel('PHONE FALL DATA FILE.xls', 'Phone_FALL_DATA', usecols = cols_fall, index_col = None)	
#data_xl_fall.to_csv('fall_data.csv',  encoding='utf-8', index=False)	

#cols_acc = [1,2]

#data_xl_acc = pd.read_excel('ACCIDENTAL DATA FILE.xlsx', 'Sheet1', usecols = cols_acc, index_col = None)
#data_xl_acc.to_csv('acc_data.csv', encoding = 'utf-8', index = False)

#creating the dataset 
data_set = list()

with open('fall_data.csv', 'r') as csvFile:
	reader = csv.reader(csvFile)
	count = 0
	for row in reader:
		if count == 0 : 
			count += 1
		else : 
			x_data = float(row[0])
			y_data = float(row[1])
			data_set.append((x_data,y_data,0))
csvFile.close()
with open('acc_data.csv', 'r') as csvFile:
	reader = csv.reader(csvFile)
	count = 0
	for row in reader:
		if count == 0 : 
			count += 1
		else : 
			x_data = float(row[0])
			y_data = float(row[1])
			data_set.append((x_data,y_data,1))
csvFile.close()

random.shuffle(data_set)

#creating data and labels 
features = list()
labels = list() 

for row in data_set: 
	features.append((row[0],row[1]))
	labels.append(row[2])

#split the data (90% train, 5% test, 5% validation)
train_data = features[:int(len(features)*0.9)]
train_labels = labels[:int(len(features)*0.9)]
test_data = features[int(len(features)*0.9):int(len(features)*0.95)]
test_labels = labels[int(len(features)*0.9):int(len(features)*0.95)]
valid_data = features[int(len(features)*0.95):]
valid_labels = labels[int(len(features)*0.95):]

#loading and training the model 
model = GaussianNB()
model.fit(train_data, train_labels)

#model metrics 
pred_labels = model.predict(test_data)
accuracy = metrics.accuracy_score(test_labels, pred_labels)
print("Accuracy : " , accuracy)







