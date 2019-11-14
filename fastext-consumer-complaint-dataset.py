# read csv
df = pd.read_csv('data/Consumer_Complaints.csv')

df.columns
selected = df[['Product','Consumer complaint narrative']]
# preprocessing

selected = selected[pd.notnull(df['Product'])]
selected = selected[pd.notnull(df['Consumer complaint narrative'])]

#  plot distribution

selected.groupby('Product').count()
plt.figure(figsize=(15, 8))
g = sns.countplot(x='Product' ,data=selected, palette="colorblind")
plt.xticks(rotation=45)


# histogram of num of characters
plt.figure(figsize=(15, 8))
sns.distplot(selected['complain_words']);

selected = selected.loc[(selected['complain_words'] > 6) & (selected['complain_words'] < 1000)]
selected_raw = selected.loc[(selected['complain_words'] > 6) & (selected['complain_words'] < 1000)]

selected.shape

# plot histogram again
plt.figure(figsize=(15, 8))
sns.distplot(selected['complain_words']);

# function to create labels
# labels = product
# text = compalint_narritive
def create_label(row,column):
    temp = ''
    text = row[column].lower()
    text = text.replace(', ', ',')
    for label in text.split(','):
        label = label.replace(' ', '-')
        lebel_t = '__label__' + label
        if temp == '':
            temp = temp + lebel_t
        else:
            temp = temp + ' ' + lebel_t
    return temp



# add new columns as labels

selected['label']  = selected.apply (lambda row: create_label(row,'Product'), axis=1)
selected_raw['label']  = selected_raw.apply (lambda row: create_label(row,'Product'), axis=1)

# function to clean data

import string
import re

def clean_text(row):
    text = row['Consumer complaint narrative']
    tt = text.replace('\n','')
    tt = tt.replace('\t','')
    tt = re.sub('[^A-Za-z0-9 ]+', '', tt)
    return tt



# cleaning text
selected['complaint']  = selected.apply (lambda row: clean_text(row), axis=1)



final_raw = selected_raw[['label','Consumer complaint narrative']]
final_clean = selected[['label','Consumer complaint narrative']]

final_raw.columns = ['label','complaint']
final_clean.columns = ['label','complaint']


# write data to files, raw and cleaned

# write to text files clean and raw data formatted to fastText input format
for index, row in final_raw.iterrows():
    text = row['label'] + ' ' + row['complaint']
    with open('data_consumer_raw.txt', 'a') as the_file:
        the_file.write(text + '\n')


for index, row in final_clean.iterrows():
    text = row['label'] + ' ' + row['complaint']
    with open('data_consumer.txt', 'a') as the_file:
        the_file.write(text + '\n')



# running same models on consumer data as in fastText tutorial
replication_result_consumer = []

raw_data_train = 'consumer_raw.train'
raw_data_valid = 'consumer_raw.valid'

clean_data_train = 'consumer.train'
clean_data_valid = 'consumer.valid'


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
replication_result_consumer.append(['Multi-label classification',recall,'Recall'])


# plotting results

result_df = pd.DataFrame.from_records(replication_result_consumer)
result_df.columns = ['wordNgrams','Value','Measure']
plt.figure(figsize=(15,7))
ax = sns.lineplot(x='wordNgrams', y='Value', hue='Measure',data=result_df)
plt.xticks(rotation=45)
plt.show()



### ngram 1-5 without lr
n_val = list(range(1,6))
all_results = []
for i in n_val:
    print(i)
    model = fasttext.train_supervised(input="consumer.train", epoch=25,wordNgrams=i)
    result = model.test("consumer.valid")
    pcc = result[1]
    recall = result[2]
    all_results.append([i,pcc,'Precision'])
    all_results.append([i,recall,'Recall'])

result_df = pd.DataFrame.from_records(all_results)
result_df.columns = ['wordNgrams','Value','Measure']
plt.figure(figsize=(15,7))
ax = sns.lineplot(x='wordNgrams', y='Value', hue='Measure',data=result_df)
plt.show()


### ngram 1-5 with lr = 1.0

n_val = list(range(1,6))
all_results = []
for i in n_val:
    print(i)
    model = fasttext.train_supervised(input="consumer.train",lr=1.0, epoch=25,wordNgrams=i)
    result = model.test("consumer.valid")
    pcc = result[1]
    recall = result[2]
    all_results.append([i,pcc,'Precision'])
    all_results.append([i,recall,'Recall'])

result_df = pd.DataFrame.from_records(all_results)
result_df.columns = ['wordNgrams','Value','Measure']
plt.figure(figsize=(15,7))
ax = sns.lineplot(x='wordNgrams', y='Value', hue='Measure',data=result_df)
plt.show()

### ngram 1-5 with lr = 1.0 & hierarichal softmax

n_val = list(range(1,6))
all_results = []
for i in n_val:
    print(i)
    model = fasttext.train_supervised(input="consumer.train",lr=1.0, epoch=25,wordNgrams=i, loss='hs')
    result = model.test("consumer.valid")
    pcc = result[1]
    recall = result[2]
    all_results.append([i,pcc,'Precision'])
    all_results.append([i,recall,'Recall'])

result_df = pd.DataFrame.from_records(all_results)
result_df.columns = ['wordNgrams','Value','Measure']
plt.figure(figsize=(15,7))
ax = sns.lineplot(x='wordNgrams', y='Value', hue='Measure',data=result_df)
plt.show()
