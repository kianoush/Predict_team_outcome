

"""
Predict home team outcome in all international soccer (football) matches
"""

import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn as nn

#print(os.listdir())

raw_data = pd.read_csv('soccer_international_history_dataset.csv')

#print(raw_data.columns)
# n = 0
# d = []
# for title in (raw_data.columns):
#     d.append(pd.unique(raw_data[title]))
#     print(d)



titles_Need_to_change = ['match_city', 'match_type']
for titles in titles_Need_to_change:
    i = 0
    for item in np.unique(raw_data[titles]):
        raw_data[titles].replace((item), (i), inplace=True)
        i += 1


all_match_country = {}

unique, counts = np.unique(raw_data['home_country'], return_counts=True)
all_match_country = dict(zip(unique, counts))

#print(raw_data.info())
country_change = ['match_country', 'away_country', 'home_country']
row_need_to_clean_away_country = []
for country in country_change:
    clean_row_away_country = []
    for item in raw_data[country]:
        if item not in unique:
            clean_row_away_country.append(item)
    clean_row_away_country = np.unique(clean_row_away_country)
    #print(clean_row_away_country)


    for item in clean_row_away_country:
        row = np.where(raw_data[country] == item)
        for i in row:
            for j in i:
                    row_need_to_clean_away_country.append(j)

row_need_to_clean_away_country = np.unique(row_need_to_clean_away_country)
for i in row_need_to_clean_away_country:
    raw_data = raw_data.drop(i)

#print(raw_data.info())


for titles in country_change:
    for item in unique:
        raw_data[titles].replace((item), (all_match_country[item]), inplace=True)

df = pd.DataFrame({'match_country': raw_data['match_country'], 'match_city': raw_data['match_city'],'match_type': raw_data['match_type'], 'home_score': raw_data['home_score'],
                   'away_score':raw_data['away_score'], 'away_country':raw_data['away_country'], 'home_country':raw_data['home_country']})

label = raw_data.iloc[:,8]
label.replace(('Loss', 'Draw', 'Win'), (0, 1, 2), inplace=True)

#print(np.where(df["away_country"] == 'Western Australia'))
pd.to_numeric(df['away_country'])

pd.to_numeric(df['match_country'])


x_train,  x_test, y_train, y_test = train_test_split(df, label, shuffle=True, test_size=0.2)
x_train,  x_valid, y_train, y_valid = train_test_split(x_train, y_train, shuffle=True, test_size=0.2)



x_train = torch.tensor(x_train.values).float()
x_test = torch.tensor(x_test.values).float()
x_valid = torch.tensor(x_valid.values).float()

y_train = torch.tensor(y_train.values)
y_test = torch.tensor(y_test.values)
y_valid = torch.tensor(y_valid.values)

Sample_num = x_train.shape[1]
Class_num = 2
Hiddenl = 10


"Model"
model = nn.Sequential(nn.Linear(Sample_num, Hiddenl),
                      nn.ReLU(),
                      nn.Linear(Hiddenl, Class_num),
                      nn.Sigmoid())
"Loss"
loss = nn.MSELoss()

"Optimizer"
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


"Train"
num_sample_train = torch.tensor(x_train.shape[0])
num_sample_test = torch.tensor(x_test.shape[0])
num_sample_valid = torch.tensor(x_valid.shape[0])
epochs = 200

for epoch in range (epochs):

    optimizer.zero_grad()
    yp = model(x_train)
    loss_value = loss(yp, y_train)

    yp_acc = torch.sum(torch.max(loss_value, 1)[1] == y_train)
    acc_train = (yp_acc.item() / num_sample_train) * 100

    loss_value.backward()
    optimizer.step()

    yp_valid = model(x_valid)
    yp_valid_acc = torch.sum(torch.max(loss_value, 1)[1] == y_valid)
    acc_valid = (yp_acc.item() / num_sample_train) * 100

    print('Epoch: ', epoch, 'loss: ', loss_value.item, 'Train acc: ', acc_train.item(), 'Valid acc: ', acc_valid.item())


print("End!!!")