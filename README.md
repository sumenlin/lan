# [under construction] Lan
This repository provides implementations of the paper: Lan: Learning to Augment Noise Tolerance for Self-report Survey Labels.
The whole code base is under construction. Please feel free to contact me if you have any questions.

## Code usages
Step 1: preparation

The model need data inputs stored in a csv file, which contains ids and dates as keys. 

Step 2: generating the feature embedding

Running the model file "emb_model" in the folder "embedding"

Parameters:
- num_epochs: the number of epoches for embedding training
- batch_size: batch size
- seed: random seed
- hid_dim: the number of dimensions of the expected embeddings
- window: the length of the time window (days) for training the embedding
- input_dim: the number of features in the original data 
- data_file: the path for the original data

Step 3: training the learning model [more updates will come after publish]

Running the model file "main_parameter" in the folder "learning"


## Notion
- The default keys of the data file are ids and dates.
- By default, the codes are running on gpu.
- The categories of labels are 0-based.
