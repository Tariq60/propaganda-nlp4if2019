import pickle
from collections import defaultdict

from input import get_tagged_corpus
from output import get_gold_from_dir
from typing import List
from nltk.tokenize import word_tokenize
from pathlib import Path

from flair.data import Sentence, Token, Corpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, CharacterEmbeddings, WordEmbeddings, BertEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


# 1. get the corpus
tokenizer = word_tokenize

corpus, id_dict = get_tagged_corpus('data/FLC/', tokenizer, mask_propaganda=False)

gold_train_dict = get_gold_from_dir('data/FLC/train/', '.task3.labels')
gold_dev_dict = get_gold_from_dir('data/FLC/dev/', '.task3.labels')
gold_test_dict = get_gold_from_dir('data/FLC/test/', '.task3.labels')

print(corpus,'\n',id_dict.keys())

# 2. what tag do we want to predict?
tag_type = 'label'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)


# 4. initialize embeddings
embedding_types: List[TokenEmbeddings] = [

#     WordEmbeddings('glove'),
    WordEmbeddings('data/urban_glove200_ohe.gensim'),

    # comment in this line to use character embeddings
    CharacterEmbeddings(),

    # comment in these lines to use flair embeddings
    FlairEmbeddings('news-forward'),
    FlairEmbeddings('news-backward'),
    
    # comment in these lines to use Bert embeddings
    BertEmbeddings('bert-base-cased'),

]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

# 6. initialize trainer
trainer: ModelTrainer = ModelTrainer(tagger, corpus)

from torch.optim import Adam
#Adam
#print("Train with adam optimzer")
#trainer: ModelTrainer = ModelTrainer(tagger, corpus, optimizer=Adam)
#lr = 0.001 #Admin Learning rate

#SGD
print("Train with SGD optimizer")
trainer: ModelTrainer = ModelTrainer(tagger, corpus, id_dict, gold_train_dict, gold_dev_dict, gold_test_dict)
lr = 0.1 #SGD Learning Rate

# 7. start training
exp_folder = 'resources/model/'
trainer.train(exp_folder,
              learning_rate=lr,
              mini_batch_size=32, #has to be <=8 if running on local GPU
              max_epochs=300,
              embeddings_in_memory=True)


# 8. plot training curves (optional)
from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_training_curves(exp_folder+'loss.tsv')
plotter.plot_weights(exp_folder+'weights.txt')