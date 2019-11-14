# FastText-implementation

- FastTextTutotial : https://fasttext.cc/docs/en/supervised-tutorial.html
Data Set: https://dl.fbaipublicfiles.com/fasttext/data/cooking.stackexchange.tar.gz


- Wikipedia Dataset: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/

- Consumer Complaint Dataset: https://catalog.data.gov/dataset/consumer-complaint-database


- Download the data sets into the same repo as the scripts
- Run the following commands in terminal after cleaning the dataset to split the data

    head -n 12404 cooking.stackexchange.txt > cooking.train
    tail -n 3000 cooking.stackexchange.txt > cooking.valid

    head -n 104121 data_consumer_raw.txt > consumer_raw.train
    tail -n 26030 data_consumer_raw.txt > consumer_raw.valid

    head -n 104121 data_consumer.txt > consumer.train
    tail -n 26030 data_consumer.txt > consumer.valid
