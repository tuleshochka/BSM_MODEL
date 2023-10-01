import pandas as pd
from sklearn import preprocessing
import pickle
import numpy as np
import os

print("Please enter the path to file containing data about flight schedule")
path_to_dataset = input()

try:
    dataset = pd.read_csv(path_to_dataset)
except:
    print("There was an error reading the file. Maybe, you have provided the wrong path")
    exit()

#if dataset['t_st'] type is object
i_id = dataset['i_id']
if 'i_id' in dataset.columns:
    dataset = dataset.drop(['i_id'], axis=1)
dataset['t_st'] = pd.to_datetime(dataset['t_st'], 
format = '%Y-%m-%d %H:%M:%S', 
errors = 'coerce')

dataset['t_st_year'] = dataset['t_st'].dt.year
dataset['t_st_month'] = dataset['t_st'].dt.month
dataset['t_st_day'] = dataset['t_st'].dt.day
dataset['t_st_hour'] = dataset['t_st'].dt.hour
dataset['t_st_minute'] = dataset['t_st'].dt.minute
dataset['t_st_second'] = dataset['t_st'].dt.second
dataset = dataset.drop(['t_st'], axis=1)

le = preprocessing.LabelEncoder()
le.fit(dataset['departure_terminal'])
dataset['departure_terminal'] = le.transform(dataset['departure_terminal'])

le.fit(dataset['checkin_terminal'])
dataset['checkin_terminal'] = le.transform(dataset['checkin_terminal'])

le.fit(dataset['airline_grouped_hash'])
dataset['airline_grouped_hash'] = le.transform(dataset['airline_grouped_hash'])

le.fit(dataset['cco_hash'])
dataset['cco_hash'] = le.transform(dataset['cco_hash'])


le.fit(dataset['flt_hash'])
dataset['flt_hash'] = le.transform(dataset['flt_hash'])

le.fit(dataset['m_city_rus1'])
dataset['m_city_rus1'] = le.transform(dataset['m_city_rus1'])

le.fit(dataset['m_city_rus2'])
dataset['m_city_rus2'] = le.transform(dataset['m_city_rus2'])

print("Please enter the path to file containing data about history of BSM")
path_to_dataset_2 = input()

import csv

import codecs

try:
    dataset_2 = pd.read_csv(path_to_dataset_2)
except:
    print("There was an error reading the file. Maybe, you have provided the wrong path")
    exit()

listA = []
with codecs.open(path_to_dataset, 'r', "utf_8_sig") as file:
    reader = csv.reader(file)
    for line in reader:
        print(line[0])
        if line[0] != 'i_id':
            listA.append(line[0])

listB = []
with codecs.open(path_to_dataset_2, 'r', "utf_8_sig") as file:
    reader = csv.reader(file)
    for line in reader:
        if line[0] != 'ident_bag':
            listB.append(line[1])

listC = []
for i in range(len(listA)):
    sum = 0
    for j in range(len(listB)):
        print(listA[i])
        print(listB[j])
        if listA[i] == listB[j]:
            sum += 1
    listC.append(sum)

print(listC)

dict = {'history_bsm': listC}
df = pd.DataFrame(dict)
df.to_csv('BSM.csv', index=False)
# path = 'BSM.csv'
# with codecs.open(path, 'r', "utf_8_sig") as file:
#     writer = csv.writer(file)
#     for i in range(len(listC)):
#         writer.writerow([listC[i]])


BSM = pd.read_csv('BSM.csv')

dataset['history_bsm'] = BSM

print(dataset.head())

# load the model from disk
loaded_model = pickle.load(open('LR_model.sav', 'rb'))

prediction = loaded_model.predict(dataset)
prediction = (np.ceil(prediction)).astype(int)

DF1 = pd.DataFrame(prediction, columns = ['Количество BSM в данном рейсе'])

DF = i_id.to_frame()
DF = DF.join(DF1['Количество BSM в данном рейсе'])
DF= DF.rename(columns={"i_id": "Идентификатор рейса"})
DF.to_csv('./output_table.csv', index=False)