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



# corpus, id_dict = get_tagged_corpus(data_folder, tokenizer, mask_propaganda=False)
# checkpoint_path = 'resources/model/best-model.pt'
# trainer: ModelTrainer = ModelTrainer.load_from_checkpoint(Path(checkpoint_path), 'SequenceTagger', corpus)
# test_data = InputParser(data_dir + 'test/', tokenizer, challenge='PTR', data_file_ending='.txt',
#                          mask_propaganda=mask_propaganda)

# test_sentences, test_id_list = read_column_data(test_data.data, {0: 'text', 1: 's_span', 2: 'e_span', 3: 'label'})

# ModelTrainer._evaluate_sequence_tagger_to_file(trainer.model, test_sentences, id_dict=test_id_list)


checkpoint_path = 'resources/model/best-model.pt'
model = SequenceTagger.load(checkpoint_path)

# test = open('test.tsv').readlines()
test = open('test.tsv').readlines()
test = [line.split('\t')[0] for line in test]

predictions : List[Sentence] = []

for i,sent in enumerate(test):
    sentence = Sentence(sent, use_tokenizer=True)
    model.predict(sentence)
    predictions.append(sentence)
    
    if i % 100 == 0:
        print('tagged {} examples'.format(i))

# pickle.dump(predictions,open('test.pkl','wb'))
# predictions = pickle.load(open('test.pkl','rb'))

submission_file = open('/datasets/test/test_submission_file.txt').readlines()
char_before = 0
doc_tags = defaultdict(list)
for line1, line2, sent in zip(submission_file, predictions, test):
    doc_id, sent_id, _ = line1.split('\t')
    if sent_id == '1':
        char_before = 0
    
    spans = line2.get_spans('label')
    if len(spans) > 0:
        doc_tags[doc_id].append((char_before, sent_id, line2.to_original_text(), spans))
    
    char_before += len(sent)
    
with open('flc-submission.txt','w') as writer:
    for key in sorted(doc_tags.keys()):
        for item in doc_tags[key]:
            char_before, sent_id, sentence, spans = item
            first_span, last_span = spans[0], spans[-1]

            label = spans[0].tag
            start_char = char_before + spans[0].start_pos
            end_char = char_before + spans[-1].end_pos
            
            writer.write('{}\t{}\t{}\t{}\n'.format(key, label, start_char, end_char))

