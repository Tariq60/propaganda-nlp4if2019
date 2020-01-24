import sys
import itertools
import numpy as np
import random as rn
import matplotlib.pyplot as plt
import torch
from pytorch_pretrained_bert import BertModel
from torch import nn
from torchnlp.datasets import imdb_dataset
from pytorch_pretrained_bert import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from IPython.display import clear_output



## Reading the data
train = open('train.tsv').readlines()
train_texts = [line.split('\t')[0] for line in train]
train_labels = [line.rstrip().split('\t')[1] for line in train]

dev = open('dev.tsv').readlines()
dev_texts = [line.split('\t')[0] for line in dev]
dev_labels = [line.rstrip().split('\t')[1] for line in dev]

print(len(train_texts), len(train_labels), len(dev_texts), len(dev_labels))

## Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
train_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:128], train_texts))
test_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:128], dev_texts))
train_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))
test_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, test_tokens))
# padding seq to be of equal lengths
train_tokens_ids = pad_sequences(train_tokens_ids, maxlen=128, truncating="post", padding="post", dtype="int")
test_tokens_ids = pad_sequences(test_tokens_ids, maxlen=128, truncating="post", padding="post", dtype="int")

train_y = np.array(train_labels, dtype=int)
test_y = np.array(dev_labels, dtype=int)
print(train_y.shape, test_y.shape)

# masking words for bert model
train_masks = [[float(i > 0) for i in ii] for ii in train_tokens_ids]
test_masks = [[float(i > 0) for i in ii] for ii in test_tokens_ids]


#######################################
## Baseline logistic regression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

baseline_model = make_pipeline(CountVectorizer(ngram_range=(1,3)), LogisticRegression()).fit(train_texts, train_labels)
baseline_predicted = baseline_model.predict(dev_texts)
with open('LR_baseline_predictions.txt','w') as f:
    for pred in baseline_predicted:
        f.write('{}\n'.format(pred))
#######################################


#######################################
## BERT model
class BertBinaryClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(BertBinaryClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, 1)
    
    def forward(self, tokens, masks=None):
        _, pooled_output = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)
        linear_output = self.linear(pooled_output)
        return linear_output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('training mode: '+device)
bert_clf = BertBinaryClassifier()
bert_clf = bert_clf.cuda()

BATCH_SIZE = 8
EPOCHS = 3

train_tokens_tensor = torch.tensor(train_tokens_ids)
train_y_tensor = torch.tensor(train_y.reshape(-1, 1)).float()
test_tokens_tensor = torch.tensor(test_tokens_ids)
test_y_tensor = torch.tensor(test_y.reshape(-1, 1)).float()

train_masks_tensor = torch.tensor(train_masks)
test_masks_tensor = torch.tensor(test_masks)
print('cuda memory allocated:'+ str(torch.cuda.memory_allocated(device)/1000000 ) + 'M')

train_dataset = TensorDataset(train_tokens_tensor, train_masks_tensor, train_y_tensor)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

test_dataset = TensorDataset(test_tokens_tensor, test_masks_tensor, test_y_tensor)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)

optimizer = Adam(bert_clf.parameters(), lr=3e-6)
loss_func = nn.BCEWithLogitsLoss().cuda()

# bert training
losses = []
steps = []
step = 0
for epoch_num in range(EPOCHS):
    bert_clf.train()
    train_loss = 0
    for step_num, batch_data in enumerate(train_dataloader):
        token_ids, masks, labels = tuple(t.to(device) for t in batch_data)
        probas = bert_clf(token_ids, masks)
        
        batch_loss = loss_func(probas, labels)
        train_loss += batch_loss.item()
        
        
        bert_clf.zero_grad()
        batch_loss.backward()
        

        clip_grad_norm_(parameters=bert_clf.parameters(), max_norm=1.0)
        optimizer.step()
        
        clear_output(wait=True)
        print('Epoch: ', epoch_num + 1)
        print("{0}/{1} loss: {2} ".format(step_num, len(train) / BATCH_SIZE, train_loss / (step_num + 1)))
        losses.append(batch_loss.item())
        steps.append(step)
        step += 1

# saving model:
with open('bert_config.json', 'w') as f:
    f.write(bert_clf.config.to_json_string())


# bert prediction
def sigmoid(x):                                        
    return 1 / (1 + np.exp(-x))
bert_clf.eval()
bert_predicted = []
all_logits = []
with torch.no_grad():
    for step_num, batch_data in enumerate(test_dataloader):

        token_ids, masks, labels = tuple(t.to(device) for t in batch_data)

        probas = bert_clf(token_ids, masks)
        numpy_probas = probas.cpu().detach().numpy()
        
        bert_predicted += list(sigmoid(numpy_probas[:, 0]) > 0.5)
#######################################

