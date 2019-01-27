import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from seqeval.metrics import f1_score
import numpy as np
from tqdm import tqdm
import math

# :: Bert parameter ::
MAX_LEN = 75
bs = 32
FULL_FINETUNING = True
BERT_model='bert-base-cased'


# :: ConLL reader ::
def conll_reader(filepath):
    all_sentences = []
    all_labels = []

    sentence_tokens = []
    sentence_labels = []

    for line in open(filepath):
        line = line.strip()

        if len(line) == 0:
            if len(sentence_tokens) > 0:
                all_sentences.append(sentence_tokens)
                all_labels.append(sentence_labels)

                sentence_tokens = []
                sentence_labels = []
            continue

        split = line.split()
        sentence_tokens.append(split[0])
        sentence_labels.append(split[-1])

    if len(sentence_tokens) > 0:
        all_sentences.append(sentence_tokens)
        all_labels.append(sentence_labels)

    return all_sentences, all_labels

def bert_tokenize(tokenizer, sentences, sentences_labels):
    all_tokenized_sentences = []
    all_tokenized_labels = []

    for sentence_idx in range(len(sentences)):
        sentence = sentences[sentence_idx]
        labels = sentences_labels[sentence_idx]

        tokenized_sentence = []
        tokenized_labels = []

        for token_idx in range(len(sentence)):
            token = sentence[token_idx]
            subtokens = tokenizer.tokenize(token)
            sublabels = [labels[token_idx]]
            if len(subtokens) > 1:
                sublabels += ['X'] * (len(subtokens)-1)

            tokenized_sentence += subtokens
            tokenized_labels += sublabels

        all_tokenized_sentences.append(tokenized_sentence)
        all_tokenized_labels.append(tokenized_labels)
    return all_tokenized_sentences, all_tokenized_labels


datapath = 'data/conll2003_ner'
train_sentences, train_labels = conll_reader(datapath+"/train.txt")
dev_sentences, dev_labels = conll_reader(datapath+"/test.txt")
test_sentences, test_labels = conll_reader(datapath+"/test.txt")

tags_vals = set(['X'])
for sentence_labels in train_labels+dev_labels+test_labels:
    for label in sentence_labels:
        tags_vals.add(label)
tags_vals = list(tags_vals)
tag2idx = {t: i for i, t in enumerate(tags_vals)}


print(":: CoNLL Dataset ::")
print(train_sentences[0:2])
print(train_labels[0:2])
print(tags_vals)
print(tag2idx)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

tokenizer = BertTokenizer.from_pretrained(BERT_model, do_lower_case=False)
train_tokenized, train_labels = bert_tokenize(tokenizer, train_sentences, train_labels)
dev_tokenized, dev_labels = bert_tokenize(tokenizer, dev_sentences, dev_labels)
test_tokenized, test_labels = bert_tokenize(tokenizer, test_sentences, test_labels)

# Sanity check: Subtoken list is as long as label list
for idx in range(len(train_tokenized)):
    assert(len(train_tokenized[idx]) == len(train_labels[idx]))

for idx in range(len(dev_tokenized)):
    assert(len(dev_tokenized[idx]) == len(dev_labels[idx]))

for idx in range(len(test_tokenized)):
    assert(len(test_tokenized[idx]) == len(test_labels[idx]))


print(list(zip(train_tokenized[4], train_labels[4])))

print("Max train len:", max([len(tokens) for tokens in train_tokenized]))
print("Max test len:", max([len(tokens) for tokens in dev_tokenized]))
print("Max test len:", max([len(tokens) for tokens in test_tokenized]))

print("Train longer MAX_LEN:", sum([int(len(tokens) > MAX_LEN) for tokens in train_tokenized]), "of", len(train_tokenized))
print("Dev longer MAX_LEN:", sum([(len(tokens) > MAX_LEN) for tokens in dev_tokenized]), "of", len(dev_tokenized))
print("Test longer MAX_LEN", sum([(len(tokens) > MAX_LEN) for tokens in test_tokenized]), "of", len(test_tokenized))

train_input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in train_tokenized], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
dev_input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in dev_tokenized], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
test_input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in test_tokenized], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

train_tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in train_labels], maxlen=MAX_LEN, value=tag2idx["O"], padding="post", dtype="long", truncating="post")
dev_tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in dev_labels], maxlen=MAX_LEN, value=tag2idx["O"], padding="post", dtype="long", truncating="post")
test_tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in test_labels], maxlen=MAX_LEN, value=tag2idx["O"], padding="post", dtype="long", truncating="post")

train_attention_masks = [[float(i>0) for i in ii] for ii in train_input_ids]
dev_attention_masks = [[float(i>0) for i in ii] for ii in dev_input_ids]
test_attention_masks = [[float(i>0) for i in ii] for ii in test_input_ids]



train_inputs = torch.tensor(train_input_ids)
dev_inputs = torch.tensor(dev_input_ids)
test_inputs = torch.tensor(test_input_ids)

train_tags = torch.tensor(train_tags)
dev_tags = torch.tensor(dev_tags)
test_tags = torch.tensor(test_tags)

train_masks = torch.tensor(train_attention_masks)
dev_masks = torch.tensor(dev_attention_masks)
test_masks = torch.tensor(test_attention_masks)


train_data = TensorDataset(train_inputs, train_masks, train_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

dev_data = TensorDataset(dev_inputs, dev_masks, dev_tags)
dev_sampler = SequentialSampler(dev_data)
dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=bs)

test_data = TensorDataset(test_inputs, test_masks, test_tags)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=bs)

model = BertForTokenClassification.from_pretrained(BERT_model, num_labels=len(tag2idx))

if torch.cuda.is_available():
    model.cuda()


if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)




def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


epochs = 20
max_grad_norm = 1.0


def evaluate(dataloader, total, desc):
    # VALIDATION on validation set
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions, true_labels = [], []
    for batch in tqdm(dataloader, total=total, desc=desc):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps

    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
    pred_subtags = [tags_vals[p_i] for p in predictions for p_i in p]
    valid_subtags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]

    # Remove X tags
    pred_tags = []
    valid_tags = []
    for idx in range(len(valid_subtags)):
        pred_subtag = pred_subtags[idx]
        valid_subtag = valid_subtags[idx]
        if valid_subtag=='X':
            continue
        valid_tags.append(valid_subtag)
        pred_tags.append(pred_subtag if pred_subtag!='X' else 'O')  # Replace 'X' tags by 'O' tags

    print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))

for epoch in range(epochs):
    print("Epoch:", epoch+1)
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in tqdm(enumerate(train_dataloader), total=math.ceil(len(train_input_ids)/bs), desc="Train"):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # forward pass
        loss = model(b_input_ids, token_type_ids=None,
                     attention_mask=b_input_mask, labels=b_labels)
        # backward pass
        loss.backward()
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        model.zero_grad()

    # print train loss per epoch
    print("Train loss: {}".format(tr_loss / nb_tr_steps))

    print("\n:: Dev Data :: ")
    evaluate(dev_dataloader, math.ceil(len(dev_input_ids) / bs), "Dev")

    print("\n:: Test Data :: ")
    evaluate(test_dataloader, math.ceil(len(test_input_ids) / bs),  "Test")

    print("\n\n")
