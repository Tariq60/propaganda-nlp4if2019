import os



def convert_file(source_path, destination, word_position=0, label_position=3, o='O', b='B'):

    with open(source_path, 'r') as f:

        sentence = ""
        document = ""
        data_set = ""
        label_doc = ''
        doc_sentence_counter = 0
        doc_id = 0
        # char_position = 0
        start_position = 0
        end_position = 0

        current_label = o



        for line in f:

            parts = line.split()

            if len(parts) == 0:

                if current_label.startswith(b):
                    label_doc += str(doc_id) + "\t" + str(start_position) + "\t" + str(
                        end_position) + "\t" + current_label + "\n"

                if sentence != '':

                    if len(document) != 0:
                        document += '  '

                    document += sentence

                    if doc_sentence_counter >= 5:
                        data_set += document + "\t" + str(doc_id) + "\n"
                        document = ""

                        with open(destination + str(doc_id) + '_docs', 'w') as f:
                            f.write(data_set)
                        with open(destination + str(doc_id) + '_labels', 'w') as f:
                            f.write(label_doc)

                        doc_id += 1
                        doc_sentence_counter = 0


                        data_set = ""
                        label_doc = ''


                    doc_sentence_counter += 1

                sentence = ''

            else:
                word = parts[word_position]
                label = parts[label_position]

                word_length = len(word)

                if len(sentence) != 0:
                    sentence += " "
                    # char_position += 1

                sentence_length = len(document) + len(sentence)
                if len(document) > 0:
                    sentence_length += 2

                if label.startswith(b):
                    if current_label.startswith(b):
                        label_doc += str(doc_id) + "\t" + str(start_position) + "\t" + str(
                            end_position) + "\t" + current_label + "\n"

                    start_position = sentence_length
                    end_position = start_position + word_length
                    current_label = label
                elif not label.startswith(o):
                    end_position = sentence_length + word_length
                else:
                    if current_label != o:
                        label_doc += str(doc_id) + "\t" + str(start_position) + "\t" + str(end_position) + "\t" + current_label + "\n"
                    current_label = o

                sentence += word

    # if len(sentence) != 0:
    #     if len(document) != 0:
    #         document += '  '
    #     document += sentence
    if len(document) != 0:
        data_set += document + "\t" + str(doc_id)

    with open(source_path + str(doc_id) + '_docs', 'w') as f:
        f.write(data_set)
    with open(source_path + str(doc_id) + '_labels', 'w') as f:
        f.write(label_doc)

    return data_set, label_doc

if __name__ == '__main__':
    convert_file('data/conll2003_ner/train.txt', 'data/conll2003_datathon/train/')
    convert_file('data/conll2003_ner/test.txt', 'data/conll2003_datathon/test/')
    convert_file('data/conll2003_ner/test.txt', 'data/conll2003_datathon/test/')
    print('done')


