import numpy as np
import pandas as pd
import re
import gc
import sys
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model, Model
from keras.layers import Embedding, LSTM, Dense, Dropout, RepeatVector, TimeDistributed, Input
from keras.callbacks import EarlyStopping
import tensorflow
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

INF = 1e9
PATH_TO_MODEL = sys.argv[1]

# from google.colab import drive
# drive.mount('/content/gdrive')

def load_LSTM_weights(is_en=False):
    if is_en:
        model_path = './final-lm/Euro_LM_final.hdf5'
    else:
        model_path = './final-lm/News_LM_final.hdf5'
    model = load_model(model_path)
    lstm_weights = model.layers[1].get_weights()
    return lstm_weights

def load_data(corpus, extension):
    CORPUS_FOLDER = '../ted-talks-corpus/'
    start_token = ''
    if extension != 'en':
        start_token = 'startTok '
    end_token = ''
    if extension != 'en':
        end_token = ' endTok'
    with open(CORPUS_FOLDER + f'train.{extension}', 'r') as f:
        lines = [line.strip()  + end_token for line in f]
        corpus["train_sentences"] = lines.copy()
        corpus["X_train"] = [start_token + re.sub('\W', ' ', line.lower()).strip() for line in lines]
        corpus["y_train"] = [re.sub('\W', ' ', line.lower()).strip() for line in lines]
    with open(CORPUS_FOLDER + f'test.{extension}', 'r') as f:
        lines = [start_token + line.strip() + end_token for line in f]
        corpus["test_sentences"] = lines.copy()
        corpus["X_test"] = [re.sub('\W', ' ', line.lower()).strip() for line in lines]
    with open(CORPUS_FOLDER + f'dev.{extension}', 'r') as f:
        lines = [line.strip() + end_token.lower() for line in f]
        corpus["X_valid"] = [start_token.lower() + re.sub('\W', ' ', line.lower()).strip() for line in lines]
        corpus["y_valid"] = [re.sub('\W', ' ', line.lower()).strip() for line in lines]
    return corpus

def read_corpora():
    # Note that model parameter is set only when finetuning
    # Arrive at maxlen for both (LM idea helps a bit)
    en_corpus = {"X_train": [], "y_train": [], "X_test": [], 
                  "y_test": [], "X_valid": [], "y_valid": [], 
                  "maxlen": 6, "vsize": 0,
                  "tokenizer": [], "model": [],
                  "train_sentences": [], "test_sentences": []}
    fr_corpus = en_corpus.copy()
    en_corpus = load_data(en_corpus.copy(), 'en')
    fr_corpus = load_data(fr_corpus.copy(), 'fr')
    return en_corpus, fr_corpus

en_corpus, fr_corpus = read_corpora()

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
        data[f'X_{s}'] = np.asarray(pad_sequences(data[f'X_{s}'], maxlen=data['maxlen'], padding='post', truncating='post'))
        if s != 'test':
            data[f'y_{s}'] = [tokenizer.texts_to_sequences([text])[0] for text in data[f'y_{s}']]
            data[f'y_{s}'] = np.asarray(pad_sequences(data[f'y_{s}'], maxlen=data['maxlen'], padding='post', truncating='post'))
    return data

en_corpus = get_seq_from_text(en_corpus.copy(), 8000)
fr_corpus = get_seq_from_text(fr_corpus.copy(), 8000)

def load_batch_data(sequences_in, sequences_out, sequences_target, steps_per_epoch, batch_size, maxlen_in, maxlen_out):
    curr_batch_num = 0
    while True:
        if curr_batch_num >= steps_per_epoch:
            curr_batch_num = 0
            continue
        curr_inputs = sequences_in[(curr_batch_num * batch_size):((curr_batch_num + 1) * batch_size)]
        curr_outputs = sequences_out[(curr_batch_num * batch_size):((curr_batch_num + 1) * batch_size)]
        curr_targets = sequences_target[(curr_batch_num * batch_size):((curr_batch_num + 1) * batch_size)]
        yield ([np.array(curr_inputs), np.asarray(curr_outputs)], np.asarray(curr_targets))
        curr_batch_num += 1

def define_translation_model(fr_corpus, en_corpus, embedding_dim=300, lstm_units=530, finetune=False):
    input_en = Input(shape=(None,))
    en_emb = Embedding(en_corpus["vsize"], embedding_dim, mask_zero=True)(input_en)
    enc_out, enc_h, enc_c = LSTM(lstm_units, return_state=True)(en_emb)

    input_fr = Input(shape=(None,))
    fr_emb = Embedding(fr_corpus["vsize"], embedding_dim, mask_zero=True)(input_fr)
    dec_out, dec_h, dec_c  = LSTM(lstm_units, return_sequences=True, return_state=True)(fr_emb, initial_state=[enc_h, enc_c])
    fin_dec_out = Dense(fr_corpus["vsize"], activation='softmax')(dec_out)

    model = Model([input_en, input_fr], fin_dec_out)
    if finetune:
        en_weights, fr_weights = load_LSTM_weights(is_en=True), load_LSTM_weights(is_en=False)
        model.layers[4].set_weights(en_weights)
        model.layers[5].set_weights(fr_weights)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.005))
    # print(model.summary())
    return model

def train_model(_model, en_corpus, fr_corpus, name):
    checkpoint = ModelCheckpoint(filepath=f'/content/gdrive/My Drive/Nlp-assign3/{name}.hdf5', 
                                 monitor="val_loss", save_best_only=True, verbose=1)
    BATCH_SIZE, model = 256, _model

    train_steps_per_epoch = np.ceil(len(en_corpus["X_train"]) / BATCH_SIZE)
    train_data_gen = load_batch_data(en_corpus["X_train"], fr_corpus["X_train"], fr_corpus["y_train"], train_steps_per_epoch, BATCH_SIZE, en_corpus["maxlen"], fr_corpus["maxlen"])
    valid_steps_per_epoch = np.ceil(len(en_corpus["X_valid"]) / BATCH_SIZE)
    valid_data_gen = load_batch_data(en_corpus["X_valid"], fr_corpus["X_valid"], fr_corpus["y_valid"], valid_steps_per_epoch, BATCH_SIZE, en_corpus["maxlen"], fr_corpus["maxlen"])
    model.fit_generator(train_data_gen, epochs=6, steps_per_epoch=train_steps_per_epoch,
                        validation_data=valid_data_gen, validation_steps=valid_steps_per_epoch,
                        verbose = 1, callbacks=[checkpoint])
    return model

def transform_seq(fr_corpus, enc_model, dec_model, inp):
    curr_st = enc_model.predict(inp)
    out_seq = [fr_corpus["tokenizer"].word_index['starttok']]
    for _ in range(fr_corpus["maxlen"]):
        target_seq = np.array(pad_sequences([out_seq], maxlen=fr_corpus["maxlen"], padding='post', truncating='post'))
        dec_tokens, dec_h, dec_c = dec_model.predict([target_seq] + curr_st)
        out_seq.append(np.argmax(dec_tokens[0, -1, :]))
        curr_st = [dec_h, dec_c]
    return fr_corpus["tokenizer"].sequences_to_texts([out_seq[1:]])[0]

def get_bleu_score(sentences_tokens, references):
    references_tokens = [[text_to_word_sequence(ref.lower())] for ref in references]
    sent_bleu_scores = [sentence_bleu(references_tokens[i], sentences_tokens[i]) for i in range(len(sentences_tokens))]
    corpus_bleu_score = corpus_bleu(references_tokens, sentences_tokens)
    return corpus_bleu_score, sent_bleu_scores

def store_bleu_score_results(en_corpus, fr_corpus, enc_model, dec_model, _t=1):
    train_references = fr_corpus["train_sentences"][:2000]
    test_references = fr_corpus["test_sentences"][:2000]
    tr_corpus_score, tr_sent_scores = get_bleu_score([transform_seq(fr_corpus, enc_model, dec_model, np.array([inp])).split() for inp in tqdm(en_corpus['X_train'][:2000])], train_references)
    test_corpus_score, test_sent_scores = get_bleu_score([transform_seq(fr_corpus, enc_model, dec_model, np.array([inp])).split() for inp in tqdm(en_corpus['X_test'][:2000])], test_references)

    with open(f'./2019111009_MT{_t}_train.txt', 'w') as _f:
        print(tr_corpus_score, file=_f)
        for i in range(len(train_references)):
            print(train_references[i].strip() + '\t' + str(tr_sent_scores[i]), file=_f)
    with open(f'./2019111009_MT{_t}_test.txt', 'w') as _f:
        print(test_corpus_score, file=_f)
        for i in range(len(test_references)):
            print(test_references[i].strip() + '\t' + str(test_sent_scores[i]), file=_f)

def create_inference_models(model):
    enc_inp = model.input[0]
    enc_emb = model.layers[2](enc_inp)
    enc_out, st_h_enc, st_c_enc = model.layers[4](enc_emb)
    enc_model = Model(enc_inp, [st_h_enc, st_c_enc])

    dec_inp = model.input[1]
    dec_emb = model.layers[3](dec_inp)
    dec_st_inp_h = Input(shape=(None,))
    dec_st_inp_c = Input(shape=(None,))
    dec_out, st_h_dec, st_c_dec = model.layers[5](dec_emb, initial_state=[dec_st_inp_h, dec_st_inp_c])
    dec_out = model.layers[6](dec_out)
    dec_model = Model([dec_inp, dec_st_inp_h, dec_st_inp_c], [dec_out, st_h_dec, st_c_dec])
    
    return enc_model, dec_model

# model = define_translation_model(fr_corpus, en_corpus)
# model = train_model(model, en_corpus, fr_corpus, 'translation_model')
model = load_model(PATH_TO_MODEL)

enc_model, dec_model = create_inference_models(model)
# store_bleu_score_results(en_corpus, fr_corpus, enc_model, dec_model)

# model = define_translation_model(fr_corpus, en_corpus, finetune='ft' in PATH_TO_MODEL)
# model = train_model(model, en_corpus, fr_corpus, enc_model, dec_model, 'translation_model')

# enc_model, dec_model = create_inference_models(model)
# store_bleu_score_results(en_corpus, fr_corpus, enc_model, dec_model, _t=2)

# prompt part - obtain translation on runtime enter CTRL + C or type '-1' on prompt to exit
while True:
    sentence = input("input sentence: ")
    if sentence == '-1':
        break
    translation = transform_seq(fr_corpus, enc_model, dec_model, np.array([en_corpus["tokenizer"].texts_to_sequences([sentence])[0]]))
    print(translation)