from transformers import BertTokenizer, BertForMaskedLM, BertForTokenClassification, BertConfig
import torch
import pandas
import os
import numpy as np
import nltk
import random
from tqdm import tqdm

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))+1e-8
    
modelConfig = BertConfig.from_pretrained('/home/share/wangyunxiao/BERT/bert-base-uncased-config.json')
tokenizer = BertTokenizer.from_pretrained('/home/share/wangyunxiao/BERT/bert-base-uncased-vocab.txt')
mask_model = BertForMaskedLM.from_pretrained('bert-base-uncased', config = modelConfig).eval()
#class_model = BertForTokenClassification.from_pretrained('bert-base-uncased', config = modelConfig).eval()

noun_tag_list = ['NN','NNS','NNP','NNPS']
verb_tag_list = ['VB','VBD','VBG','VBN','VBP','VBZ']

path=os.path.join("/home/share/wangyunxiao/Charades/caption/charades_test.csv")
df=pandas.read_csv(path)
description=df['description']
neg_index_list =[]
neg_token_list = []

for i in tqdm(range(len(description))):
    sentence = description[i]
    words = tokenizer.tokenize(sentence)
    tagged = nltk.pos_tag(words)
    token_list = []
    idx_list = []
    for j,(word,tag) in enumerate(tagged):
        token_list.append(word)
        if tag in verb_tag_list and word != 'is':
            idx_list.append(j)
    
    if len(idx_list) == 0:
        for j,(word,tag) in enumerate(tagged):
            if tag in noun_tag_list and word != 'person':
                idx_list.append(j)

    if len(idx_list) == 0:
        print(i,tagged)
        idx_list.append(0)

    index = random.choice(idx_list)

    input_ids = tokenizer(token_list, return_tensors="pt", is_pretokenized=True)["input_ids"]
    outputs = mask_model(input_ids)
    prediction_scores = outputs[0][0][index+1]
    q_sample = prediction_scores.detach().numpy()
    # pred = np.argsort(-sample, axis=0)[:30] # 降序排列
    # sent = tokenizer.convert_ids_to_tokens(pred)
    # print(sent)
    
    token_list[index] = '[MASK]'    
    input_ids = tokenizer(token_list, return_tensors="pt", is_pretokenized=True)["input_ids"]
    outputs = mask_model(input_ids)
    prediction_scores = outputs[0][0][index+1]
    sample = prediction_scores.detach().numpy()
    pred = np.argsort(-sample, axis=0)[:35] # 降序排列
    o_sample = sample[pred]
    q_sample = q_sample[pred]
    o_sample = softmax(o_sample)
    q_sample = softmax(q_sample)
    r_sample = o_sample/(q_sample+1e-8)

    r_pred = np.argsort(-r_sample, axis=0)[:25]

    # sent = tokenizer.convert_ids_to_tokens(pred)
    # print(sent)

    sent = tokenizer.convert_ids_to_tokens(pred[r_pred])
    neg_index_list.append(index)
    neg_token_list.append(sent)
    # print(sent)

df['neg_index'] = neg_index_list
df['neg_word'] = neg_token_list

df.to_csv('charades_test_gen.csv',index=False)


    