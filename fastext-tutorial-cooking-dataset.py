import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import fasttext


# replicating the fastext tutorial code

replication_result = []
# model on raw data
model = fasttext.train_supervised(input="cooking-r.train")
result = model.test("cooking-r.valid")
pcc = result[1]
recall = result[2]
replication_result.append(['Raw Data',pcc,'Precision'])
replication_result.append(['Raw Data',recall,'Recall'])

# model on Preprocessed Data

model = fasttext.train_supervised(input="cooking.train")
result = model.test("cooking.valid")
pcc = result[1]
recall = result[2]
replication_result.append(['Pre Processed Data',pcc,'Precision'])
replication_result.append(['Pre Processed Data',recall,'Recall'])

# more epochs and larger learning rate

model = fasttext.train_supervised(input="cooking.train", lr=1.0, epoch=25)
result = model.test("cooking.valid")
pcc = result[1]
recall = result[2]
replication_result.append(['epochs & Learning Rate',pcc,'Precision'])
replication_result.append(['epochs & Learning Rate',recall,'Recall'])

#word n-grams¶
model = fasttext.train_supervised(input="cooking.train", lr=1.0, epoch=25, wordNgrams=2)
result = model.test("cooking.valid")
pcc = result[1]
recall = result[2]
replication_result.append(['epochs, Learning Rate & n-gram 2',pcc,'Precision'])
replication_result.append(['epochs, Learning Rate & n-gram 2',recall,'Recall'])

# using loss function hierarchical softmax

model = fasttext.train_supervised(input="cooking.train", lr=1.0, epoch=25, wordNgrams=2, bucket=200000, dim=50, loss='hs')
result = model.test("cooking.valid")
pcc = result[1]
recall = result[2]
replication_result.append(['hierarchical softmax & n-gram 2',pcc,'Precision'])
replication_result.append(['hierarchical softmax & n-gram 2',recall,'Recall'])

# using Multi-label classification

model = fasttext.train_supervised(input="cooking.train", lr=0.5, epoch=25, wordNgrams=2, bucket=200000, dim=50, loss='ova')
result = model.test("cooking.valid")
pcc = result[1]
recall = result[2]
replication_result.append(['Multi-label classification',pcc,'Precision'])
replication_result.append(['Multi-label classification',recall,'Recall'])

# making plots of results

replication_result_df = pd.DataFrame.from_records(replication_result)
replication_result_df.columns = ['Model','Value','Measure']
plt.figure(figsize=(15,7))
ax = sns.lineplot(x='Model', y='Value', hue='Measure',data=replication_result_df)
plt.xticks(rotation=45)
plt.show()


# using 1-5 ngram value with lr = 1 on default data fastext
n_val = list(range(1,6))
all_results_orignal = []
for i in n_val:
    print(i)
    model = fasttext.train_supervised(input="cooking.train",lr=1.0, epoch=25,wordNgrams=i)
    result = model.test("cooking.valid")
    pcc = result[1]
    recall = result[2]
    all_results_orignal.append([i,pcc,'Precision'])
    all_results_orignal.append([i,recall,'Recall'])


# plot results

result_df = pd.DataFrame.from_records(all_results_orignal)
result_df.columns = ['wordNgrams','Value','Measure']
plt.figure(figsize=(15,7))
ax = sns.lineplot(x='wordNgrams', y='Value', hue='Measure',data=result_df)
plt.show()



# using 1-5 ngram value with no lr

n_val = list(range(1,6))
all_results_orignal = []
for i in n_val:
    print(i)
    model = fasttext.train_supervised(input="cooking.train", epoch=25,wordNgrams=i)
    result = model.test("cooking.valid")
    pcc = result[1]
    recall = result[2]
    all_results_orignal.append([i,pcc,'Precision'])
    all_results_orignal.append([i,recall,'Recall'])

result_df = pd.DataFrame.from_records(all_results_orignal)
result_df.columns = ['wordNgrams','Value','Measure']
plt.figure(figsize=(15,7))
ax = sns.lineplot(x='wordNgrams', y='Value', hue='Measure',data=result_df)
plt.show()


# using 1-5 ngram value with hierricahl softmax

n_val = list(range(1,6))
all_results_orignal_hs = []
for i in n_val:
    print(i)
    model = fasttext.train_supervised(input="cooking.train",lr=1.0, epoch=25,wordNgrams=i,loss='hs')
    result = model.test("cooking.valid")
    pcc = result[1]
    recall = result[2]
    all_results_orignal_hs.append([i,pcc,'Precision'])
    all_results_orignal_hs.append([i,recall,'Recall'])

result_df = pd.DataFrame.from_records(all_results_orignal_hs)
result_df.columns = ['wordNgrams','Value','Measure']
plt.figure(figsize=(15,7))
ax = sns.lineplot(x='wordNgrams', y='Value', hue='Measure',data=result_df)
plt.show()
