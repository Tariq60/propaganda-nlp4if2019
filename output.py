import numpy as np
import os


def sentences_to_pred_output(sentence, doc_id, doc_dict, predictions=None):

    # output_string = ''

    # if doc_id in  ['767129999', '999001032', '779394730', '7625464289'] or doc_id in [767129999, 999001032, 779394730, 7625464289]:
    #     print('here')

    # for i in range(len(sentences)):

        # #TODO this is not the doc_id
        # doc_id = i
        #
        # sentence = sentences[i]

    current_label = 'O'
    start = 0
    end = 0
    for j in range(len(sentence)):

        token = sentence.tokens[j]

        if predictions is not None:
            prediction = predictions[j].value.strip()
        else:
            prediction = token.tag['label'].strip()

        start_token = int(token.tags['s_span'].value)

        end_token = int(token.tags['e_span'].value)

        if prediction != current_label:
            if current_label != 'O':
                # output_string += str(doc_id) + '\t' + str(start) + '\t' + str(end) + '\t' + prediction
                if doc_id not in doc_dict:
                    doc_dict[doc_id] = []
                if [start, end, current_label] not in doc_dict[doc_id]:
                    doc_dict[doc_id].append([start, end, current_label])
            # if prediction != 'O':
            current_label = prediction
            start = start_token
            end = end_token
        else:
            # if prediction != 'O':
            end = end_token

        if j == len(sentence)-1:
            if current_label != 'O':
                # output_string += str(doc_id) + '\t' + str(start) + '\t' + str(end_token) + '\t' + prediction
                if doc_id not in doc_dict:
                    doc_dict[doc_id] = []
                if [start, end_token, prediction] not in doc_dict[doc_id]:
                    doc_dict[doc_id].append([start, end_token, prediction])

    return doc_dict

def soft_f1_from_files(pred_file, gold_file):

    pred_doc_dict = to_dict(get_lines(pred_file))

    gold_doc_dict = to_dict(get_lines(gold_file))

    return soft_f1(pred_doc_dict, gold_doc_dict)


def soft_f1(pred_doc_dict, gold_doc_dict):

    pred_ids = [key for key,value in pred_doc_dict.items()]
    label_ids = [key for key,value in gold_doc_dict.items()]
    distinct_ids = list(set(pred_ids + label_ids))

    tp = []
    fp = []
    fn = []

    for id_ in distinct_ids:
        if id_ in pred_doc_dict:
            pred_labels = pred_doc_dict[id_]
        else:
            pred_labels = []

        if id_ in gold_doc_dict:
            gold_labels = gold_doc_dict[id_]
        else:
            gold_labels = []

        gold_indices = []

        if len(pred_labels) > 0:

            for pred_label in pred_labels:
                pred_start = int(pred_label[0])
                pred_end = int(pred_label[1])
                pred_l = pred_label[2]
                pred_span = float(pred_end - pred_start)

                pred_set = set(range(pred_start,pred_end))

                found = False

                if len(gold_labels) > 0:

                    for i in range(len(gold_labels)):

                        gold_label = gold_labels[i]

                        gold_start = int(gold_label[0])
                        gold_end = int(gold_label[1])
                        gold_l = gold_label[2]
                        gold_span = float(gold_end - gold_start)
                        gold_set = set(range(gold_start,gold_end))


                        if len(pred_set.intersection(gold_set)) > 1 and pred_l == gold_l:

                            found = True

                            tp_ = len(pred_set.intersection(gold_set)) / pred_span
                            fp_ = len(pred_set - gold_set) / pred_span
                            fn_ = len(gold_set - pred_set) / gold_span

                            tp.append(tp_)
                            fp.append(fp_)
                            fn.append(fn_)

                            gold_indices.append(i)

                if not found:
                    tp.append(0.0)
                    fp.append(1.0)
                    fn.append(0.0)

        if len(gold_labels) > 0:
            for i in range(len(gold_labels)):
                if i not in gold_indices:
                    tp.append(0.0)
                    fp.append(0.0)
                    fn.append(1.0)
    return f1(tp, fp, fn)

def precision(tp,fp):
    return np.mean(tp) / (np.mean(tp) + np.mean(fp))

def recall(tp, fn):
    return np.mean(tp) / (np.mean(tp) + np.mean(fn))

def f1(tp,fp,fn):
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    return 2 * ( (prec * rec) / (prec + rec))

def get_gold_from_dir(directory, file_ending):
    lines = []
    for filename in os.listdir(directory):
        if filename.endswith(file_ending) and filename[0] != '.':
            lines += get_lines(os.path.join(directory, filename))
    doc_dict = to_dict(lines)
    return doc_dict


def get_lines(file_path):
    lines = []
    with open(file_path, 'r') as f:
        for line in f:
            if len(line) > 0:
                lines.append(line)
    return lines

def to_dict(lines):

    doc_dict = {}

    for line in lines:
        if len(line) > 0:
            id_, label, start, end  = line.split()
            if id_ not in doc_dict:
                doc_dict[id_] = []
            doc_dict[id_].append([start, end, label])

    return doc_dict


if __name__ == '__main__':

    f1 = soft_f1_from_files('data/Conll2003_datathon/gold/pred', 'data/Conll2003_datathon/gold/gold')
    print(f1)



