# -*- coding: utf-8 -*-
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence, text
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Embedding, LSTM, Bidirectional, Conv1D, GlobalMaxPooling1D, Input, concatenate
from keras.layers import MaxPooling1D
from keras.layers import add as addition
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from keras_self_attention import SeqSelfAttention

import misc

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

if __name__ == "__main__":
    # Parameter
    num_words = 5000
    oov_token = "UNK"
    reddit_weight = 0.7
    EMBEDDING_SIZE = 300
    HIDDEN_SIZE = 300
    BATCH_SIZE = 128
    EPOCHS = 7
    dropout = 0.25
    # Anzahl der Durchläufe für durchschnittliche F1 und Acc.
    iterations = 5

    # Daten einlesen
    reddit, semeval = misc.read_lines()
    # train dev test split
    # Datensätze separat erstellen und danach nach Bedarf vereinigen
    X_train_reddit, y_train_reddit, X_dev_reddit, \
        y_dev_reddit, X_test_reddit, y_test_reddit = misc.train_dev_test(
            reddit[1], reddit[0])
    X_train_semeval, y_train_semeval, X_dev_semeval, \
        y_dev_semeval, X_test_semeval, y_test_semeval = misc.train_dev_test(
            semeval[1], semeval[0])
    # Vokabular erstellen
    tk = text.Tokenizer(num_words=num_words + 1,
                        lower=True, split=" ", oov_token=oov_token)
    # Vokabular aus beiden Trainingsdatenmengen erstellen
    tk.fit_on_texts(X_train_reddit + X_train_semeval)
    # https://github.com/keras-team/keras/issues/9637
    # https://github.com/keras-team/keras/issues/8092
    tk.word_index = {e: i for e, i in tk.word_index.items() if i <= num_words}
    tk.word_index[tk.oov_token] = num_words + 1
    # Eingabevektoren erzeugen
    # # reddit
    X_train_reddit = tk.texts_to_sequences(X_train_reddit)
    X_dev_reddit = tk.texts_to_sequences(X_dev_reddit)
    X_test_reddit = tk.texts_to_sequences(X_test_reddit)
    # # semeval
    X_train_semeval = tk.texts_to_sequences(X_train_semeval)
    X_dev_semeval = tk.texts_to_sequences(X_dev_semeval)
    X_test_semeval = tk.texts_to_sequences(X_test_semeval)
    # Polsterung
    # # reddit
    X_train_reddit = sequence.pad_sequences(X_train_reddit, maxlen=100)
    X_dev_reddit = sequence.pad_sequences(X_dev_reddit, maxlen=100)
    X_test_reddit = sequence.pad_sequences(X_test_reddit, maxlen=100)
    # # semeval
    X_train_semeval = sequence.pad_sequences(X_train_semeval, maxlen=100)
    X_dev_semeval = sequence.pad_sequences(X_dev_semeval, maxlen=100)
    X_test_semeval = sequence.pad_sequences(X_test_semeval, maxlen=100)
    # Vereinigung + Gewichtung
    X_train = np.concatenate((X_train_reddit, X_train_semeval))
    # Gewichte erzeugen für sample_weights-Parameter
    sample_weights = [reddit_weight for sample in X_train_reddit] + \
        [1.0 for sample in X_train_semeval]
    sample_weights = np.array(sample_weights)
    y_train = np.concatenate((y_train_reddit, y_train_semeval))
    X_dev = np.concatenate((X_dev_reddit, X_dev_semeval))
    y_dev = np.concatenate((y_dev_reddit, y_dev_semeval))

    # Hier werden die F1- und Acc.-Werte der besten Modelle pro Iteration gesammelt
    f1_cnn = []
    acc_cnn = []
    for i in range(iterations):
        metrics_cnn = misc.Metrics("CNN", i)

        cnn_model = Sequential()
        cnn_model.add(Embedding(num_words + 2, EMBEDDING_SIZE))
        cnn_model.add(
            Conv1D(HIDDEN_SIZE*2, 6, activation='relu',
                   use_bias=False))
        cnn_model.add(MaxPooling1D())
        cnn_model.add(Dropout(dropout))
        cnn_model.add(
            Conv1D(200, 3, activation='relu',
                   use_bias=False))
        cnn_model.add(GlobalMaxPooling1D())
        cnn_model.add(Dropout(dropout))
        cnn_model.add(Dense(128, activation='relu'))
        cnn_model.add(Dropout(dropout))
        cnn_model.add(Dense(1, activation='sigmoid'))

        cnn_model.compile(
            loss='binary_crossentropy',
            optimizer='adam', metrics=['accuracy']
        )
        history = cnn_model.fit(
            X_train, y_train, batch_size=BATCH_SIZE,
            epochs=EPOCHS, validation_data=(X_dev, y_dev),
            sample_weight=sample_weights,
            callbacks=[metrics_cnn],
            verbose=0
        )
        cnn_model.summary()

        acc_cnn.append(history.history["val_acc"][metrics_cnn.ind])
        f1_cnn.append(max(metrics_cnn.val_f1s))

    f1_attn = []
    acc_attn = []
    print("F1-Werte aller Modelle: Index in Liste entspr. Index im ausgelagerten Modell")
    print("    => Modell mit dem besten Wert einlesen")
    print("CNN F1:\t", f1_cnn)

    # Standardabweichung und Durchschnitt für Accuracy und F1-Wert ausgeben
    misc.print_results(
        [
            [
                "CNN", np.std(acc_cnn), np.mean(
                    acc_cnn), np.std(f1_cnn), np.mean(f1_cnn)
            ]
        ],
        iterations
    )

    # Modelle einlesen -> bester Wert ermittelt durch Epochen
    best_cnn_model = load_model(
        "model_CNN_{}.h5".format(f1_cnn.index(max(f1_cnn))))

    # Modelle auf beiden Datensätzen auswerten (Reddit bzw. Semeval)
    # und Maße ausgeben
    print("Name & Datensatz : Score\tAccuracy\tF1-Wert\tConfusion Matrix")
    print("CNN    -> reddit : \t", *misc.eval_model(best_cnn_model, "CNN reddit",
                                                    X_test_reddit, y_test_reddit))
    print("CNN    -> semeval: \t", *misc.eval_model(best_cnn_model, "CNN semeval",
                                                    X_test_semeval, y_test_semeval))
