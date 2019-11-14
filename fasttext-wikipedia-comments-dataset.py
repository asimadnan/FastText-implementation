#loading libraries

import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#reading csv in pandas dataframe
df1 = pd.read_csv('data/wikipedia_comments.csv')

#dropping 'id' column
df1 = df1.drop(columns=['id'],axis=1)

#adding new column 'neutral'
df1['neutral'] = 0

#adding value 1 to column 'neutral' where every other rating column is = 0
df1.loc[(df1['severe_toxic']== False) &(df1['toxic']== False) &
(df1['obscene']== False) &(df1['threat']== False) &  (df1['insult']== False)
& (df1['identity_hate']== False), 'neutral'] = 1

#new column ' comments' where number of words in column 'comment_text' is stored
df1['comments']  = df1['comment_text'].str.split().str.len()

#df1 is copied for raw data to be used later
raw_data = df1

#created labels in a new column 'label' in both dataframes (i.e 'df1' & 'raw_data')
def create_label(row):
    label_text = ''
    if row['toxic'] == True:
        label_text = label_text + '__label__' + 'toxic '
    if row['severe_toxic'] == True:
        label_text = label_text + '__label__' + 'severe_toxic '
    if row['obscene'] == True:
        label_text = label_text + '__label__' + 'obscene '
    if row['threat'] == True:
        label_text = label_text + '__label__' + 'threat '
    if row['insult'] == True:
        label_text = label_text + '__label__' + 'insult '
    if row['identity_hate'] == True:
        label_text = label_text + '__label__' + 'identity_hate '
    if row['neutral'] == True:
        label_text = label_text + '__label__' + 'neutral '
    return label_text



df1['label']  = df1.apply (lambda row: create_label(row), axis=1)
raw_data['label']  = df1.apply (lambda row: create_label(row), axis=1)

#function to clean text by removing punctuations and converting all characters in lowercase
import string
import re

def clean_text(row):
    text = row['comment_text']
    tt = text.replace('\n','')
    tt = tt.replace('\t','')
    tt = re.sub('[^A-Za-z0-9 ]+', '', tt)
    return tt

# This function was only applied on 'df1' data frame so that we can compare with raw_data later
df1['comments']  = df1.apply (lambda row: clean_text(row), axis=1)

#only two columns were needed so storing them in 2 variables
final_df1 = df1[['label','comments']]
final_raw = raw_data[['label','comments']]

#writing files in txt
for index, row in final_df1.iterrows():
    text = row['label'] + ' ' + row['comments']
    with open('data_clean.txt', 'a') as the_file:
            the_file.write(text + '\n')

for index, row in final_raw.iterrows():
    text = row['label'] + ' ' + row['comments']
    with open('data_raw.txt', 'a') as the_file:
            the_file.write(text + '\n')

# running same models on wikipedia dataset as in fastText tutorial
replication_result_consumer = []

raw_data_train = 'data_wiki_raw.train'
raw_data_valid = 'data_wiki_raw.valid'

clean_data_train = 'data_wiki.train'
clean_data_valid = 'data_wiki.valid'


# simple model on raw data
model = fasttext.train_supervised(input=raw_data_train)
result = model.test(raw_data_valid)
pcc = result[1]
recall = result[2]
replication_result_consumer.append(['Raw Data',pcc,'Precision'])
replication_result_consumer.append(['Raw Data',recall,'Recall'])

#pre processed data
model = fasttext.train_supervised(input=clean_data_train)
result = model.test(clean_data_valid)
pcc = result[1]
recall = result[2]
replication_result_consumer.append(['Pre Processed Data',pcc,'Precision'])
replication_result_consumer.append(['Pre Processed Data',recall,'Recall'])

# more epochs and learning rate
model = fasttext.train_supervised(input=clean_data_train, lr=1.0, epoch=25)
result = model.test(clean_data_valid)
pcc = result[1]
recall = result[2]
replication_result_consumer.append(['epochs & Learning Rate',pcc,'Precision'])
replication_result_consumer.append(['epochs & Learning Rate',recall,'Recall'])

# n-grams
model = fasttext.train_supervised(input=clean_data_train, lr=1.0, epoch=25, wordNgrams=2)
result = model.test(clean_data_valid)
pcc = result[1]
recall = result[2]
replication_result_consumer.append(['epochs, Learning Rate & n-gram 2',pcc,'Precision'])
replication_result_consumer.append(['epochs, Learning Rate & n-gram 2',recall,'Recall'])

# hierarichal softmax
model = fasttext.train_supervised(input=clean_data_train, lr=1.0, epoch=25, wordNgrams=2, bucket=200000, dim=50, loss='hs')
result = model.test(clean_data_valid)
pcc = result[1]
recall = result[2]
replication_result_consumer.append(['hierarchical softmax & n-gram 2',pcc,'Precision'])
replication_result_consumer.append(['hierarchical softmax & n-gram 2',recall,'Recall'])

#multi label classification
model = fasttext.train_supervised(input=clean_data_train, lr=0.5, epoch=25, wordNgrams=2, bucket=200000, dim=50, loss='ova')
result = model.test(clean_data_valid)
pcc = result[1]
recall = result[2]
replication_result_consumer.append(['Multi-label classification',pcc,'Precision'])
rep
#plotting the above models simultaneously and checking precision and recall
result_df = pd.DataFrame.from_records(replication_result_consumer)
result_df.columns = ['wordNgrams','Value','Measure']
plt.figure(figsize=(15,7))
ax = sns.lineplot(x='wordNgrams', y='Value', hue='Measure',data=result_df)
plt.xticks(rotation=45)
plt.show()

#Model with ngram 1-5 without using learning rate
n_val = list(range(1,6))
all_results = []
for i in n_val:
    print(i)
    model = fasttext.train_supervised(input="data_wiki.train", epoch=25,wordNgrams=i)
    result = model.test("data_wiki.valid")
    pcc = result[1]
    recall = result[2]
    all_results.append([i,pcc,'Precision'])
    all_results.append([i,recall,'Recall'])

result_df = pd.DataFrame.from_records(all_results)
result_df.columns = ['wordNgrams','Value','Measure']
plt.figure(figsize=(15,7))
ax = sns.lineplot(x='wordNgrams', y='Value', hue='Measure',data=result_df)
plt.show()

#Model with ngram 1-5 with learning rate = 1.0
n_val = list(range(1,6))
all_results = []
for i in n_val:
    print(i)
    model = fasttext.train_supervised(input="data_wiki.train",lr=1.0, epoch=25,wordNgrams=i)
    result = model.test("data_wiki.valid")
    pcc = result[1]
    recall = result[2]
    all_results.append([i,pcc,'Precision'])
    all_results.append([i,recall,'Recall'])

result_df = pd.DataFrame.from_records(all_results)
result_df.columns = ['wordNgrams','Value','Measure']
plt.figure(figsize=(15,7))
ax = sns.lineplot(x='wordNgrams', y='Value', hue='Measure',data=result_df)
plt.show()

#model with ngram 1-5 with learning rate = 1.0 & hierarichal softmax
n_val = list(range(1,6))
all_results = []
for i in n_val:
    print(i)
    model = fasttext.train_supervised(input="data_wiki.train",lr=1.0, epoch=25,wordNgrams=i, loss='hs')
    result = model.test("data_wiki.valid")
    pcc = result[1]
    recall = result[2]
    all_results.append([i,pcc,'Precision'])
    all_results.append([i,recall,'Recall'])

result_df = pd.DataFrame.from_records(all_results)
result_df.columns = ['wordNgrams','Value','Measure']
plt.figure(figsize=(15,7))
ax = sns.lineplot(x='wordNgrams', y='Value', hue='Measure',data=result_df)
plt.show()
