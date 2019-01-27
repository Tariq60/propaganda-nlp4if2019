import numpy as np
from tqdm import *

urban_dict_file = 'data/urban_dictionary.csv'

other_embedding_file = 'data/glove/glove.6B.200d.txt'

destination = 'data/urban_glove200_ohe.txt'

lower = False

urban_dict = {}

urban_dim = None
other_dim = None

print('\n\nLoading urban embeddings')

with open(urban_dict_file, 'r') as f:

    for i, line in enumerate(tqdm(f)):
        if i==0: continue
        if len(line) < 3:
            continue
        try:
            emb_data = line.strip().split(',')
            term = emb_data[0]

            embedding = np.array([np.float64(x) for x in emb_data[1:]])

            if not urban_dim:
                urban_dim = len(embedding)

        except:
            continue
        if lower:
            term = term.lower()

        if term not in urban_dict:

            urban_dict[term] = {}
            urban_dict[term]['urban'] = []
            urban_dict[term]['other'] = []

        urban_dict[term]['urban'].append(embedding)

# for term, emb in urban_dict.items():
#     urban_dict[term] = np.mean(emb,0)

print('\n\nLoading other embeddings')

with open(other_embedding_file, 'r') as f:
    for line in tqdm(f):
        emb_data = line.split()
        term = emb_data[0]

        embedding = np.array([np.float64(x) for x in emb_data[1:]])

        if not other_dim:
            other_dim = len(embedding)

        if lower:
            term = term.lower()

        if term not in urban_dict:

            urban_dict[term] = {}
            urban_dict[term]['urban'] = []
            urban_dict[term]['other'] = []

        urban_dict[term]['other'].append(embedding)

print('\n\ngetting OHE vectors')

ohe_label_dict = {}
ohe_term_dict = {}
for i, file_name in enumerate(['data/christian_words1.csv', 'data/christian_words2.csv']):
    with open(file_name, 'r') as f:
        for line in tqdm(f):
            elements = line.split(';')
            term, label = elements[0].strip(), elements[1].strip()
            label = label + '_' + str(i)

            if lower:
                term.lower()

            if label not in ohe_label_dict:
                ohe_label_dict[label] = len(ohe_label_dict)
            if term not in ohe_term_dict:
                ohe_term_dict[term] = []
            if label not in ohe_term_dict[term]:
                ohe_term_dict[term].append(label)




print('\n\nCombining and printing embeddings')

with open(destination, 'w') as f:
    header = str(len(urban_dict)) + ' ' + str(other_dim + urban_dim + len(ohe_label_dict)) + '\n'
    f.write(header)

    for term, emb in tqdm(urban_dict.items()):
        if len(emb['urban']) == 0:
            urban_emb = np.zeros(urban_dim)
        else:
            urban_emb = np.mean(emb['urban'],0)
        if len(emb['other']) == 0:
            other_emb = np.zeros(other_dim)
        else:
            other_emb = np.mean(emb['other'], 0)

        ohe_vec = np.zeros(len(ohe_label_dict))

        if term in ohe_term_dict:
            for label in ohe_term_dict[term]:
                ohe_vec[ohe_label_dict[label]] = 1.0

        combined = np.concatenate((other_emb,urban_emb, ohe_vec))
        f.write(term + ' ' + " ".join(["%.7f" % x for x in combined]) + '\n')


print('done')






