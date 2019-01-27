from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from typing import List
import datetime
from input import get_tagged_corpus
from output import get_gold_from_dir
from nltk.tokenize import word_tokenize
from flair.trainers import ModelTrainer
from flair_trainer import ModelTrainer
from pathlib import Path
from input import InputParser , read_column_data

# tokenizer = word_tokenize
#
# data_dir = 'data/testing_1/'
# mask_propaganda = False
#
# test_data = InputParser(data_dir + 'test/', tokenizer, challenge='PTR', data_file_ending='.txt',
#                          mask_propaganda=mask_propaganda)
#
# test_sentences, test_id_list = read_column_data(test_data.data, {0: 'text', 1: 's_span', 2: 'e_span', 3: 'label'})



# CoNLL 2003 NER dataset
columns = {0: 'text', 1: 'pos', 2: 'np', 3: 'ner'}
# data_folder = 'data/datasets-v3_1/tasks-2-3/'
data_folder = 'data/datasets-v3_1/new2/'

exp_folder = "resources/taggers/new_nomask-urban-glove-ohe-ner-%s/" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
print(exp_folder)

#Load corpus
# corpus: TaggedCorpus = NLPTaskDataFetcher.load_column_corpus(data_folder, columns,
#                                                               train_file='train.txt',
#                                                               test_file='test.txt',
#                                                               dev_file='test.txt',
#                                                               tag_to_biloes='ner')

tokenizer = word_tokenize

corpus, id_dict = get_tagged_corpus(data_folder, tokenizer, mask_propaganda=False)

gold_train_dict = get_gold_from_dir(data_folder + 'train/', '.task3.labels')
gold_dev_dict = get_gold_from_dir(data_folder + 'test/', '.task3.labels')
gold_test_dict = get_gold_from_dir(data_folder + 'test/', '.task3.labels')

print(corpus)

# 2. what tag do we want to predict?
# tag_type = 'ner'
tag_type = 'label'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# 4. initialize embeddings
embedding_types: List[TokenEmbeddings] = [

    # WordEmbeddings('glove'),
    WordEmbeddings('data/urban_glove200_ohe.gensim'),

    # comment in this line to use character embeddings
    # CharacterEmbeddings(),

    # comment in these lines to use flair embeddings
    FlairEmbeddings('news-forward'),
    FlairEmbeddings('news-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

# 6. initialize trainer

from torch.optim import Adam

#Adam
#print("Train with adam optimzer")
#trainer: ModelTrainer = ModelTrainer(tagger, corpus, optimizer=Adam)
#lr = 0.001 #Admin Learning rate

#SGD
print("Train with SGD optimizer")
# trainer: ModelTrainer = ModelTrainer(tagger, corpus, id_dict, gold_train_dict, gold_dev_dict, gold_test_dict)

# checkpoint_path = 'resources/taggers/example-ner-test/final-model.pt'
# checkpoint_path = '/home/akamath/Ash2/datathon-flair/resources/taggers/new_nomask-urban-glove-ohe-ner-2019-01-25 13:01/best-model.pt'
# checkpoint_path = 'resources/taggers/best_model/final-model.pt'
checkpoint_path = 'resources/taggers/only_train_new_nomask-urban-glove-ohe-ner-2019-01-26 17:49/best-model.pt'

trainer: ModelTrainer = ModelTrainer.load_from_checkpoint(Path(checkpoint_path), 'SequenceTagger', corpus)


# data_dir = 'data/datasets-v3_1/new2/'
# data_dir = 'data/testing_1/'
data_dir = 'data/final/'
mask_propaganda = False

test_data = InputParser(data_dir + 'test/', tokenizer, challenge='PTR', data_file_ending='.txt',
                         mask_propaganda=mask_propaganda)

test_sentences, test_id_list = read_column_data(test_data.data, {0: 'text', 1: 's_span', 2: 'e_span', 3: 'label'})

ModelTrainer._evaluate_sequence_tagger_to_file(trainer.model, test_sentences, id_dict=test_id_list)

# lr = 0.1 #SGD Learning Rate
#
#
# # 7. start training
# trainer.train(exp_folder,
#               learning_rate=lr,
#               mini_batch_size=32,
#               max_epochs=300,
#               embeddings_in_memory=True)
#
# # 8. plot training curves (optional)
# from flair.visual.training_curves import Plotter
# plotter = Plotter()
# plotter.plot_training_curves(exp_folder+'/loss.tsv')
# plotter.plot_weights(exp_folder+'/weights.txt')