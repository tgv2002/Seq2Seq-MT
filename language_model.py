import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import tensorflow
import keras.backend as K
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import sys

INF = 1e9
PATH_TO_MODEL = sys.argv[1]
# from google.colab import drive
# drive.mount('/content/gdrive')

def read_data(folder_path, ext_name, maxlen):
    corpus = {"X_train": [], "y_train": [], "X_test": [], 
              "y_test": [], "X_valid": [], "y_valid": [], 
              "maxlen": maxlen, "vsize": 0, 
              "tokenizer": [], "model": [],
              "train_sentences": [], "test_sentences": []}
    with open(folder_path + f'train.{ext_name}', 'r') as f:
        lines = [line.strip() for line in f]
        corpus["train_sentences"] = ['startTok ' + line + ' endTok' for line in lines]
        corpus["X_train"] = ['startTok ' + re.sub('\W', ' ', line.lower()).strip() + ' endTok' for line in lines]
    with open(folder_path + f'test.{ext_name}', 'r') as f:
        lines = [line.strip() for line in f]
        corpus["test_sentences"] = ['startTok ' + line + ' endTok' for line in lines]
        corpus["X_test"] = ['startTok ' + re.sub('\W', ' ', line.lower()).strip() + ' endTok' for line in lines]
    with open(folder_path + f'dev.{ext_name}', 'r') as f:
        corpus["X_valid"] = ['startTok ' + re.sub('\W', ' ', line.lower()).strip() + ' endTok' for line in f]
    return corpus

def get_inputs_and_outputs(_sequences):
    inputs = [_sequence[:i+1] for _sequence in _sequences for i in range(len(_sequence) - 1)]
    outputs = [_sequence[i+1] for _sequence in _sequences for i in range(len(_sequence) - 1)]
    return inputs, outputs

def get_seq_from_text(data, num_words):
    # Keras tokenizer - fit on text data
    tokenizer = Tokenizer(num_words = num_words, oov_token='<UNK>')
    tokenizer.fit_on_texts(data["X_train"])
    data['tokenizer'] = tokenizer
    data['vsize'] = num_words
    splits = ["train", "test", "valid"]

    # Generating sequences from tokens for piece of text 
    for s in splits:
        data[f'X_{s}'] = [tokenizer.texts_to_sequences([text])[0] for text in data[f'X_{s}']]
    return data

def load_batch_data(sequences, steps_per_epoch, batch_size, vsize, maxlen):
    curr_batch_num = 0
    while True:
        if curr_batch_num >= steps_per_epoch:
            curr_batch_num = 0
            continue
        curr_inputs, curr_outputs = get_inputs_and_outputs(sequences[(curr_batch_num * batch_size):((curr_batch_num + 1) * batch_size)])
        curr_inputs = np.array(pad_sequences(curr_inputs, maxlen=maxlen, padding='pre'))
        curr_outputs = to_categorical(curr_outputs, num_classes=vsize)
        yield (np.array(curr_inputs), np.asarray(curr_outputs))
        curr_batch_num += 1

def define_LSTM_LM(vsize, maxlen, embedding_dim=300, lstm_units=530):
    model = Sequential()
    model.add(Embedding(vsize, embedding_dim, input_length=maxlen))
    model.add(LSTM(lstm_units, return_sequences=False))
    model.add(Dense(vsize, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))
    print(model.summary())
    return model

def train_model(corpus, name, epochs=10):
    checkpoint = ModelCheckpoint(filepath=f'/content/gdrive/My Drive/Nlp-assign3/{name}.hdf5', 
                                 monitor="val_loss", save_best_only=True, verbose=1)
    SENT_PER_BATCH = 50

    train_steps_per_epoch = np.ceil(len(corpus["X_train"]) / SENT_PER_BATCH)
    train_data_gen = load_batch_data(corpus["X_train"], train_steps_per_epoch, SENT_PER_BATCH, corpus["vsize"], corpus["maxlen"])
    valid_steps_per_epoch = np.ceil(len(corpus["X_valid"]) / SENT_PER_BATCH)
    valid_data_gen = load_batch_data(corpus["X_valid"], valid_steps_per_epoch, SENT_PER_BATCH, corpus["vsize"], corpus["maxlen"])

    corpus["model"].fit_generator(train_data_gen, epochs=epochs, steps_per_epoch=train_steps_per_epoch,
                                          validation_data=valid_data_gen, validation_steps=valid_steps_per_epoch,
                                          verbose = 1, callbacks=[checkpoint])
    return corpus

def calculate_perplexity(_text, corpus):
    text = re.sub('\W', ' ', _text.lower()).strip()
    tokens = corpus["tokenizer"].texts_to_sequences([text])[0]
    x, y = [tokens[:i+1] for i in range(len(tokens) - 1)], np.array([tokens[i+1] for i in range(len(tokens) - 1)])
    x, y = np.array(pad_sequences(x, maxlen=corpus["maxlen"], padding='pre')), np.asarray(to_categorical(y, num_classes=corpus['vsize']))
    try:
        cross_entropy = corpus["model"].evaluate(x, y, verbose=3)
        perplexity = np.exp(cross_entropy)
        probability = 1 / (perplexity ** y.shape[0])
        return perplexity, probability
    except Exception as e:
        return INF, 0

def store_perplexity_results(corpus, s=''):
    train_perplexity_scores = [calculate_perplexity(sentence, corpus)[0]
                        for sentence in corpus['train_sentences']]
    average_train_perplexity = sum(train_perplexity_scores) / len(train_perplexity_scores)
    with open(f'./2019111009_LM_train{s}.txt', 'w') as _f:
        print(average_train_perplexity, file=_f)
        for i in range(len(corpus['train_sentences'])):
            print(corpus['train_sentences'][i].strip() + '\t' + str(train_perplexity_scores[i]), file=_f)
    test_perplexity_scores = [calculate_perplexity(sentence, corpus)[0]
                        for sentence in corpus['test_sentences']]
    average_test_perplexity = sum(test_perplexity_scores) / len(test_perplexity_scores)
    with open(f'./2019111009_LM_test{s}.txt', 'w') as _f:
        print(average_test_perplexity, file=_f)
        for i in range(len(corpus['test_sentences'])):
            print(corpus['test_sentences'][i].strip() + '\t' + str(test_perplexity_scores[i]), file=_f)

# euro_corpus = read_data('/content/gdrive/My Drive/Nlp-assign3/europarl-corpus/', 'europarl', maxlen=11)
euro_corpus = read_data('../europarl-corpus/', 'europarl', maxlen=11)
# news_corpus = read_data('/content/gdrive/My Drive/Nlp-assign3/news-crawl-corpus/', 'news', maxlen=8)
news_corpus = read_data('../news-crawl-corpus/', 'news', maxlen=8)
# print("Europarl train: ", euro_corpus["X_train"][:10])
# print("News train: ", news_corpus["X_train"][:10])

fin_euro_corpus = get_seq_from_text(euro_corpus.copy(), 5000)
fin_news_corpus = get_seq_from_text(news_corpus.copy(), 5000)

# fin_euro_corpus["model"] = define_LSTM_LM(fin_euro_corpus["vsize"], fin_euro_corpus["maxlen"])
# fin_news_corpus["model"] = define_LSTM_LM(fin_news_corpus["vsize"], fin_news_corpus["maxlen"])

# fin_euro_corpus = train_model(fin_euro_corpus.copy(), 'Euro_LM_final')
# fin_news_corpus = train_model(fin_news_corpus.copy(), 'News_LM_final')

fin_euro_corpus["model"] = load_model(PATH_TO_MODEL)
# fin_euro_corpus["model"] = load_model('/content/gdrive/My Drive/Nlp-assign3/Euro_LM.hdf5')

# print(calculate_perplexity("you have requested a debate on this subject", fin_euro_corpus))
# print(calculate_perplexity("Ces derniers sont donc en cage, tout comme le chauffeur.", fin_news_corpus))

# store_perplexity_results(fin_euro_corpus)
# store_perplexity_results(fin_news_corpus, '2')

# prompt part - obtain probability on runtime enter CTRL + C or type '-1' on prompt to exit
while True:
    sentence = input("input sentence: ")
    if sentence == '-1':
        break
    probability = calculate_perplexity(sentence, fin_euro_corpus)[1]
    print(probability)