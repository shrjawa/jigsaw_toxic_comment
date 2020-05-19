import pandas as pd
import tensorflow as tf
import numpy as np
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from sklearn.linear_model import LogisticRegression

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-cased')  # tensorflow in use

df1=pd.read_csv(r"D:\jigsaw-kaggle\jigsaw-toxic-comment-train-processed-seqlen128.csv")
#df1=df1.iloc[:1200,1:3]
df1=df1.iloc[:,1:3]
lables=df1.iloc[:,1]
nonzero_lables= lables.to_numpy().nonzero()
df2=[df1.iloc[nonzero_lables[0][i],:] for i in nonzero_lables[0][:500]]
df3=[]
for i in range(500):
    if(i not in nonzero_lables[0]):
        df3.append(df1.iloc[i,:])



input_ids=df1.iloc[:,0].apply(lambda x:tf.constant(tokenizer.encode(x,max_length=125)))
len_list=[len(i) for i in input_ids]  #redundant here 
max_len=max(len_list)

padded=np.array([np.append(i,np.array([0]*(125-len(i)))) for i in input_ids.values]) # make all ids of same length
attention_mask = np.where(padded != 0.0, 1.0, 0.0) # to stop model from reading extra zeros

padded=padded.astype(int)  #int32 or int64 is necessary

with tf.device("/device:CPU:0"):
    output=model(padded,attention_mask=attention_mask)   

lables=df1.iloc[:,1]
features = output[0][:,:].numpy()

clf = LogisticRegression()
r_clf.fit(output[:1000], lables[:1000])

clf.score(test_features, test_labels)

