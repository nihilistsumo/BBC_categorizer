import argparse, json, spacy, random

import keras
from keras.utils import to_categorical
import keras.backend as K
from keras.models import Model
from keras.layers import Activation
from keras.layers import Embedding, Input, TimeDistributed, Dropout
from keras.layers import LSTM, Lambda, concatenate, Dense, Bidirectional
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier

import numpy as np
from pathos.threading import ThreadPool

def prepare_data(data_dict, split_docids, elmo_embeds, classes_list):
    X = []
    y = []
    random.shuffle(split_docids)
    for docid in split_docids:
        X.append(elmo_embeds[()][docid])
        y.append(classes_list.index(data_dict[docid]['cat']))
    X = np.array(X)
    y = to_categorical(y, len(classes_list))
    return X, y

def dense_categorizer(Xtrain, ytrain, Xval, yval, optim, num_classes, embed_vec_len=3072, learning_rate=0.01, num_epochs=3,
                      num_bacthes=1, pat=10):
    doc_vec = Input(shape=(embed_vec_len,), dtype='float32', name='docvec')
    drop = Dropout(0.5)
    dense_layer1 = Dense(1024, activation='relu', input_shape=(embed_vec_len,), kernel_regularizer=regularizers.l2(0.01))
    d1_out = dense_layer1(drop(doc_vec))
    dense_layer2 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))
    d2_out = dense_layer2(d1_out)
    dense_layer3 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))
    d3_out = dense_layer3(d2_out)
    dense_layer4 = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))
    d4_out = dense_layer4(d3_out)
    class_prob_out = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(d4_out)

    model = Model(inputs=[doc_vec], outputs=[class_prob_out])

    if optim == 'adadelta':
        opt = keras.optimizers.Adadelta(lr=learning_rate, clipnorm=1.25)
    else:
        opt = keras.optimizers.Adam(lr=learning_rate)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=pat)
    model.summary()

    history = model.fit(Xtrain, ytrain, validation_data=(Xval, yval), epochs=num_epochs,
                        batch_size=num_bacthes, verbose=1, callbacks=[es])
    return model


def main():
    parser = argparse.ArgumentParser(description='Categorization model for bbc dataset')
    parser.add_argument('-d', '--data_dict', required=True, help='Path to bbc data dict file')
    parser.add_argument('-s', '--split_dataset', required=True, help='Path to split data dict')
    parser.add_argument('-e', '--elmo_embed', required=True, help='Path to elmo embeddings')
    parser.add_argument('-op', '--optimizer', required=True, help='Optimizer (adadelta/adam)')
    parser.add_argument('-vl', '--vec_len', type=int, help='Length of elmo emebd vec')
    parser.add_argument('-ep', '--num_epochs', type=int, help='No. epochs')
    parser.add_argument('-b', '--num_batches', type=int, help='No. batches')
    parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate')
    parser.add_argument('-p', '--patience', type=int, help='Patience value')
    # parser.add_argument('-n', '--num_classes', type=int, help='Num of classes for categorizer')
    args = vars(parser.parse_args())
    data_dict_file = args['data_dict']
    split_data_file = args['split_dataset']
    elmo_embed_file = args['elmo_embed']
    optim = args['optimizer']
    veclen = args['vec_len']
    epochs = args['num_epochs']
    batches = args['num_batches']
    lr = args['learning_rate']
    pat = args['patience']
    # num_cl = args['num_classes']

    with open(data_dict_file, 'r') as dt:
        data_dict = json.load(dt)
    with open(split_data_file, 'r') as sd:
        split_dat = json.load(sd)
    elmo_embed = np.load(elmo_embed_file, allow_pickle=True)
    unique_classes = list(set([data_dict[d]['cat'] for d in data_dict.keys()]))

    train_doc_list = split_dat['train']
    val_doc_list = split_dat['val']
    test_doc_list = split_dat['test']

    Xtrain, ytrain = prepare_data(data_dict, train_doc_list, elmo_embed, unique_classes)
    Xval, yval = prepare_data(data_dict, val_doc_list, elmo_embed, unique_classes)
    Xtest, ytest = prepare_data(data_dict, test_doc_list, elmo_embed, unique_classes)

    m = dense_categorizer(Xtrain, ytrain, Xval, yval, optim, len(unique_classes), veclen, lr, epochs, batches, pat)
    test_eval = m.evaluate(Xtest, ytest)
    print('Evaluation on test set: '+str(test_eval-d))

if __name__ == '__main__':
    main()