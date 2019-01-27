# PIG Datathon - NER

run `train_ner.py` to train the NER classifier.
This transforms the datathon data type into a `FLAIR` dataset and trains the model. The best model will be stored after n epochs.


To predict using this trained model, change the directories in `predict.py` and run it. This creates a file in the format for the challenge

