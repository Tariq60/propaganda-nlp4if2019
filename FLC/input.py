import numpy as np
import os
import pickle
from pytorch_pretrained_bert import BertTokenizer
from typing import List, Dict
import re
from pathlib import Path

from flair.data import Sentence, Token, Corpus
from nltk.tokenize import sent_tokenize, word_tokenize

class InputParser:

    def __init__(self, directory, tokenizer, challenge='PTR', data_file_ending='_docs', label_file_ending=None, token_replacement=['##'], mask_propaganda=False):
        self.directory = directory
        self.challenge = challenge
        self.label_file_ending = label_file_ending
        self.data_file_ending = data_file_ending
        self.tokenizer = tokenizer
        self.token_replacement = token_replacement

        self.mask_propaganda = mask_propaganda

        self.documents = None
        self.data = None

        if self.challenge == 'PAL':
            self.parse_PAL()

        elif self.challenge == 'PTR':
            self.parse_PTR()

        elif self.challenge == 'PSL':
            self.parse_PSL()

        else:
            raise Exception('No such challenge implemented')

    def parse_PSL(self):
        pass

    def parse_PTR(self):
        lines = self.__loop_through_files_in_dir__(self.directory, self.data_file_ending, get_lines=False)
        self.__loop_through_sentence_lines__(lines)

        if self.label_file_ending:

            # labels = self.__load_doc_lines__((os.path.join(self.directory, self.label_file_ending)))
            labels = self.__loop_through_files_in_dir__(self.directory, self.label_file_ending)
            self.__loop_through_PTR_labels__(labels)

        self.data = []

        for data_point in self.documents:

            sentence = data_point['sentence']

            if self.label_file_ending:
                distinct_label_indices = self.__get_distinct_label_indices__(data_point)
            else:
                distinct_label_indices = [None]

            document_splits = self.__parse_and_chunk_PTR_document__(sentence, distinct_label_indices, data_point['label'])

            # document_splits = self.__parse_and_chunk_PTR_document__('Hello Jonas. What is your name?')

            data_point['splits'] = document_splits

            self.__tokenize_document_splits__(document_splits, data_point['id'])


    def __tokenize_document_splits__(self, document_splits, doc_id):
        tokenized_document_splits = []
        for s_id, split in enumerate(document_splits):
            tokenized_sentence = []
            for chunk, first, second, labels in split:
                tokens = self.tokenizer(chunk)
                for token in tokens:
                    token_length = len(token)
                    tok_rep = False
                    for rep in self.token_replacement:
                        if rep in token:
                            tok_rep = True
                            token_length -= len(rep)

                    if tok_rep:
                        tokenized_sentence.append([token, first, first+token_length, ['X']])
                    else:
                        tokenized_sentence.append([token, first, first+token_length, labels])
                    first += token_length
            # tokenized_document_splits.append(tokenized_sentence)
            self.data.append({'ds_id':str(doc_id) + '_' + str(s_id), 'd_id': doc_id, 's_id': s_id, 'sentence': tokenized_sentence})
        # return tokenized_document_splits


    def parse_PAL(self):

        lines = self.__loop_through_files_in_dir__(self.directory, self.label_file_ending)

        self.documents = []

        for line in lines:
            data_points = line.split('\t')
            self.documents.append({
                        'id': data_points[1],
                        'label':data_points[2],
                        'sentence':data_points[0] })

    def __loop_through_sentence_lines__(self, lines):

        self.documents = []
        self.id_dict = {}

        for line in lines:
            # data_points = line.split('\t')

            # if self.challenge == 'PAL':
            #     label = data_points[0].strip()
            if self.challenge == 'PTR':
                label = []
            else:
                label = None
            self.documents.append({
                        'id': line[1],
                        # 'id': data_points[1].strip(),
                        # 'sentence':data_points[0],
                        'sentence':line[0],
                        'label':label})

        self.documents = sorted(self.documents, key=lambda k: int(k['id']))
        for i in range(len(self.documents)):
            self.id_dict[self.documents[i]['id']] = i

    def __loop_through_PTR_labels__(self, labels):
        for label in labels:
            label_data = label[0].split('\t')
            id = self.id_dict[label_data[0].strip()]
            self.documents[id]['label'].append([int(label_data[2]), int(label_data[3].strip()), label_data[1].strip()])

    def __get_distinct_label_indices__(self, data_point):
        label_indices = []
        sentence = data_point['sentence']
        for labels in data_point['label']:
            label_indices += labels[:2]
        label_indices = list(set(label_indices)) + [len(sentence)]
        label_indices.sort()
        return label_indices

    def __parse_and_chunk_PTR_document__(self, sentence, distinct_label_indices=[None], data_labels=[], default_label='O'):

        document_splits = [[]]
        sentence_splits = document_splits[0]
        indices_index = 0
        previous = 0
        is_word = False

        nltk_sentences = sent_tokenize(sentence)
        nltk_sentence_pos = 0
        nltk_char_pos = 0
        nltk_end = False
        token_labels = []
        skip = False

        for i in range(len(sentence)):

            if skip:
                try:
                    if sentence[i+1] != " ":
                        skip = False
                except:
                    pass
                continue

            character = sentence[i]
            # if i in distinct_label_indices:
            #     print('h')
            # if character == '\n':
            #     continue
            if nltk_char_pos == 0:
                nltk_end = False
            try:
                if character == nltk_sentences[nltk_sentence_pos][nltk_char_pos]:
                    while True:
                        if nltk_char_pos + 1 == len(nltk_sentences[nltk_sentence_pos]):
                            nltk_end = True
                            nltk_sentence_pos += 1
                            nltk_char_pos = 0
                            break
                        else:
                            nltk_char_pos += 1

                        if nltk_sentences[nltk_sentence_pos][nltk_char_pos] != ' ':
                            break
            except:
                pass


            if distinct_label_indices[indices_index] == i:
                if is_word:
                    token_labels = []
                    for labels in data_labels:
                        if labels[0]-1 <= previous and labels[1]+2 >= i:
                            if self.mask_propaganda:
                                token_labels.append('propaganda')
                            else:
                                token_labels.append(labels[2])
                    if token_labels == []:
                        token_labels.append(default_label)
                    sentence_splits.append([sentence[previous:i], previous, i, token_labels])
                    if character == ' ':
                        is_word = False
                else:
                    is_word = True
                previous = i
                indices_index += 1

            elif character == ' ' and is_word:
                token_labels = []
                for labels in data_labels:
                    if labels[0]-1 <= previous and labels[1]+2 >= i:
                        if self.mask_propaganda:
                            token_labels.append('propaganda')
                        else:
                            token_labels.append(labels[2])
                if token_labels == []:
                    token_labels.append(default_label)
                sentence_splits.append([sentence[previous:i], previous, i, token_labels])
                is_word = False
            elif character != ' ' and not is_word:
                is_word = True
                previous = i
            elif (character == ' ' and not is_word):
                document_splits += [[]]
                sentence_splits = document_splits[len(document_splits) - 1]
            elif  (nltk_end):
                token_labels = []
                for labels in data_labels:
                    if labels[0]-1 <= previous and labels[1]+2 >= i:
                        if self.mask_propaganda:
                            token_labels.append('propaganda')
                        else:
                            token_labels.append(labels[2])
                if token_labels == []:
                    token_labels.append(default_label)
                sentence_splits.append([sentence[previous:i+1], previous, i+1, token_labels])
                is_word = False

                try:
                    if sentence[i+1] == " ":
                        skip = True
                except:
                    pass
                if i+1 < len(sentence):

                    document_splits += [[]]
                    sentence_splits = document_splits[len(document_splits) - 1]
                    # nltk_end = False

        if is_word:
            sentence_splits.append([sentence[previous:i+1], previous, i+1, token_labels])

        return document_splits

    def __loop_through_files_in_dir__(self, directory, file_ending=None, get_lines=True):
        lines = []
        for filename in os.listdir(directory):
            if filename.endswith(file_ending) and filename[0] != '.':
                lines += self.__load_doc_lines__(os.path.join(directory, filename), get_lines=get_lines)
        return lines

    def __load_doc_lines__(self, file_path, get_lines=True):
        lines = []

        doc_id = re.findall('article(\d*)', file_path)[0]
        with open(file_path, 'r') as f:
            if get_lines:
                for line in f:
                    if len(line) > 0:
                        lines.append([line.replace('\n',''), doc_id])
            else:
                lines.append([f.read().replace('\n', ' '), doc_id])

        return lines


# @staticmethod
def read_column_data(
                    # path_to_column_file: Path,
                    data_list,
                     column_name_map: Dict[int, str],
                     infer_whitespace_after: bool = True):
    """
    Reads a file in column format and produces a list of Sentence with tokenlevel annotation as specified in the
    column_name_map. For instance, by passing "{0: 'text', 1: 'pos', 2: 'np', 3: 'ner'}" as column_name_map you
    specify that the first column is the text (lexical value) of the token, the second the PoS tag, the third
    the chunk and the forth the NER tag.
    :param path_to_column_file: the path to the column file
    :param column_name_map: a map of column number to token annotation name
    :param infer_whitespace_after: if True, tries to infer whitespace_after field for Token
    :return: list of sentences
    """
    sentences: List[Sentence] = []
    #
    # try:
    #     lines: List[str] = open(str(path_to_column_file), encoding='utf-8').read().strip().split('\n')
    # except:
    #     # log.info('UTF-8 can\'t read: {} ... using "latin-1" instead.'.format(path_to_column_file))
    #     lines: List[str] = open(str(path_to_column_file), encoding='latin1').read().strip().split('\n')

    # most data sets have the token text in the first column, if not, pass 'text' as column
    text_column: int = 0
    label_column: int = 1
    for column in column_name_map:
        if column_name_map[column] == 'text':
            text_column = column
        if column_name_map[column] == 'label':
            label_column = column

    #   self.data.append({'ds_id':str(doc_id) + '_' + str(s_id), 'd_id': doc_id, 's_id': s_id, 'sentence': tokenized_sentence})
    #         # return tokenized_document_splits

    id_list = []

    # sentence: Sentence = Sentence()
    for s_list in data_list:
        sentence: Sentence = Sentence()



        for t_list in s_list['sentence']:


            # if line.startswith('#'):
            #     continue

            # if line.strip().replace('﻿', '') == '':

            # else:

            # fields: List[str] = re.split("\s+", line)



            token = Token(t_list[text_column])
            for column in column_name_map:
                if len(t_list) > column:
                    if column != text_column and column != label_column:
                        token.add_tag(column_name_map[column], str(t_list[column]))
                    elif column == label_column:
                        token.add_tag(column_name_map[column], str(t_list[column][0]))

            sentence.add_token(token)

        if len(sentence) > 0:
            id_list.append({'d_id': s_list['d_id'], 's_id': s_list['s_id'], 'ds_id': s_list['ds_id']})
            sentence.infer_space_after()
            sentences.append(sentence)

    # if len(sentence.tokens) > 0:
    #     sentence.infer_space_after()
    #     sentences.append(sentence)

    return sentences, id_list


def get_tagged_corpus(data_dir, tokenizer=None, name='propaganda', mask_propaganda=False):
    if tokenizer is None:
        bert_tokenizer = BertTokenizer('/Users/jonaspfeiffer/test/ner-bert-master/BERT/uncased_L-12_H-768_A-12/vocab.txt')
        tokenizer = bert_tokenizer.tokenize

    train_data = InputParser(data_dir + 'train/', tokenizer, challenge='PTR', data_file_ending='.txt',
                             label_file_ending='.task3.labels', mask_propaganda=mask_propaganda)
    dev_data = InputParser(data_dir + 'dev/', tokenizer, challenge='PTR', data_file_ending='.txt',
                             label_file_ending='.task3.labels', mask_propaganda=mask_propaganda)
    test_data = InputParser(data_dir + 'test/', tokenizer, challenge='PTR', data_file_ending='.txt',
                             label_file_ending='.task3.labels', mask_propaganda=mask_propaganda)

    train_sentences, train_id_list = read_column_data(train_data.data, {0: 'text', 1: 's_span', 2: 'e_span', 3: 'label'})
    dev_sentences, dev_id_list = read_column_data(dev_data.data, {0: 'text', 1: 's_span', 2: 'e_span', 3: 'label'})
    test_sentences, test_id_list = read_column_data(test_data.data, {0: 'text', 1: 's_span', 2: 'e_span', 3: 'label'})

    return Corpus(train_sentences, dev_sentences, test_sentences, name), {'train': train_id_list, 'dev': dev_id_list, 'test':test_id_list}
    # return Corpus(train_sentences), {'train': train_id_list}


if __name__ == '__main__':

    # train_data = InputParser('data/Conll2003_datathon/', tokenizer, challenge='PTR', data_file_ending ='_docs', label_file_ending='_labels')
    # # test_data = InputParser('data/PTR_Test/', tokenizer.tokenize, challenge='PTR')

    # train = read_column_data(data.data, {0:'text', 1:'s_span', 2:'e_span', 3:'label'})
    # test =

    tokenizer = word_tokenize

    tagged_corpus, id_dict = get_tagged_corpus('data/datasets-v3_1/tasks-2-3/', tokenizer)

    print('done')

