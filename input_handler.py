import sys
import numpy as np
import pandas as pd

import torch

lang_1 = str(sys.argv[0]) # himodel.vec
lang_2 = str(sys.argv[1]) # {mr,ml,bn,pa,ne,te,ta,gu}model.vec
cognate_file = str(sys.arv[2]) # file consisting of gold data

# function to read lang_1 file
def read_language_vec_file(filepath):
    '''
    input: pass the path of the bin file
    output: Return a pandas dataframe with word and vector
    '''
    print('Loading Vector File')
    size = {
        'total_examples' : int,
        'size' : int
    }
    lang_embeddings = {}
    f = open(filepath, 'r')
    for line in f:
        file_contents = line.split()
        if len(file_contents) < 5:
            size['total_examples'] = int(file_contents[0])
            size['size'] = int(file_contents[1])
        else:
            words = file_contents[0]
            embeddings = np.array([float(emb) for emb in file_contents[1:]])

            lang_embeddings[words] = embeddings

    print("Done Loading, ", size['total_examples'], " Loaded..")
    return pd.DataFrame.from_dict(lang_embeddings, orient='index').transpose(), size

# Open Cognate text file for both languages
def load_cognate_file(filepath):
    '''
    input : pass the input filepath of the text folder
    output : list of tuples containing (lang_1_words, lang_2_words, gold)
    '''
    print('Loading Data File')
    f = open(filepath, 'r')
    lang_1_words = []
    lang_2_words = []
    gold = []
    for line in f:
        file_contents = line.split(';')
        lang_1_words.append(file_contents[2])
        lang_2_words.append(file_contents[3])
        gold.append(file_contents[6])
    
    print('File Loaded and Processed!')

    return [(words_1, words_2, gold) for words_1, words_2, gold in zip(lang_1_words, 
                                                                       lang_2_words,
                                                                       gold)]

                                                        
def final_embedding_data_loader(lang_1, lang_2, cognate_file):
    
    df_1, _ = read_language_vec_file(lang_1)
    df_2, _ = read_language_vec_file(lang_2)

    cognate = load_cognate_file(cognate_file)

    final_data = {
        'embd_1' : torch.tensor,
        'embd_2' : torch.tensor,
        'gold' : int
    }

    prepared_data = []
    stacked_tensor = torch.empty([1, 2, 100])
    data_X = []
    labels = []

    for cog in cognate:
        if (cog[0] not in df_1.keys()) or (cog[1] not in df_2.keys()): 
            cognate.remove(cog)
        else:
            final_data['tensor_1'] = torch.from_numpy(df_1[cog[0]])
            final_data['tensor_2'] = torch.from_numpy(df_2[cog[1]])
            final_data['gold'] = int(cog[2])

            stacked_tensor = torch.stack(final_data['tensor_1'], final_data['tensor_2'])
            stacked_tensor = stacked_tensor.reshape(1, 2, 100)

            prepared_data.append((stacked_tensor, final_data['gold']))
            #(torch.stack(final_data['tensor_1'], final_data['tensor_2'])
            data_X.append(stacked_tensor)
            labels.append(final_data['gold'])

    final_data = pd.DataFrame.from_dict(final_data)
    
    print('Model input data prepared and processed..')

    return final_data, prepared_data, data_X, labels

