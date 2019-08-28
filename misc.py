from sklearn.model_selection import train_test_split
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras import backend as K


# Dateisätze aufspalten in jew. Datensätze
def train_dev_test(X: list, y: list, train: float = 0.6,
                   dev: float = 0.2, test=0.2) -> list:
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=1.0 - train, random_state=1)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=1)
    return X_train, y_train, X_dev, y_dev, X_test, y_test


# Modell auswerten zum Vergleich
def eval_model(model, name: str, X: list, y: list):
    score, acc = model.evaluate(X, y)
    pred = model.predict_classes(X, verbose=0)
    f1 = f1_score(y, pred)
    return score, acc, f1, confusion_matrix(y, pred)


# Create upsampled data set, shuffle to be done in shell by make
# Ready for consumption by 'read_lines'
def upsample_lines(folder="./data/", out="reddit_upsample.tmp",
                   reddit="reddit.tsv", semeval="semeval.tsv"):
    files = [reddit, semeval]
    reddit = [[], []]
    reddit_pos = [[], []]
    reddit_neg = [[], []]
    semeval = [[], []]
    for file in files:
        with open(folder + file, "r") as f:
            if file == files[0]:
                for line in f:
                    num, label, text = line.rstrip().split("\t")
                    if label == "1":
                        reddit_pos[0].append(int(label))
                        reddit_pos[1].append(text)
                    else:
                        reddit_neg[0].append(int(label))
                        reddit_neg[1].append(text)
            else:
                for line in f:
                    num, label, text = line.rstrip().split("\t")
                    semeval[0].append(int(label))
                    semeval[1].append(text)
    sampled_reddit_pos = upsample_reddits(
        reddit_pos, len(reddit_neg[0]))
    new_reddit_set = [
        sampled_reddit_pos[0] + reddit_neg[0],
        sampled_reddit_pos[1] + reddit_neg[1]
    ]

    with open(folder + out, "w") as f:
        for line in zip(new_reddit_set[0], new_reddit_set[1]):
            f.write("{}\t{}\n".format(line[0], line[1]))

    return reddit, semeval


def read_lines(folder="./data/", reddit="reddit_upsample_shuffle.tsv",
               semeval="semeval.tsv"):
    # Dateinamen in files sind bereits vorbearbeitete Daten
    # Auskommentierte Zeilen sind zum hochsampeln da
    files = [reddit, semeval]
    reddit = [[], []]
    semeval = [[], []]
    for file in files:
        with open(folder + file, "r") as f:
            if file == files[0]:
                for line in f:
                    label, text = line.rstrip().split("\t")
                    reddit[0].append(int(label))
                    reddit[1].append(text)
            else:
                for line in f:
                    num, label, text = line.rstrip().split("\t")
                    semeval[0].append(int(label))
                    semeval[1].append(text)

    return reddit, semeval


# Redditdaten hochsampeln da die Klassen unausgeglichen sind
def upsample_reddits(reddit_pos: list, neg_length: int) -> list:
    from random import randint

    addition = [[], []]

    # solange ein Unterschied in den Längen von pos. und neg. besteht
    # pos. Datenpunkte zuf. aussuchen und anhängen
    while len(addition[0]) != (neg_length - len(reddit_pos[0])):
        rnd = randint(0, len(reddit_pos[0]) - 1)
        addition[0].append(reddit_pos[0][rnd])
        addition[1].append(reddit_pos[1][rnd])

    # Unittests damit die Längen erhalten bleiben
    assert len(addition[0]) == len(addition[1]), "{} {}".format(
        len(addition[0]), len(addition[1]))
    assert len(addition[0]) == (neg_length - len(reddit_pos[0])), "{} {}".format(
        len(addition[0]), (neg_length - len(reddit_pos[0])))
    assert len(addition[1]) == (neg_length - len(reddit_pos[0]))

    # Kombinierte Datensätze zurückgeben
    return [addition[0] + reddit_pos[0], addition[1] + reddit_pos[1]]


# Werte strukturiert ausgeben
def print_results(params: list, iterations: int):
    print("Results: {} iterations".format(iterations))
    for val in params:
        print("\t{}:\tacc std: {}\tacc avg: {}\tf1 std: {}\tf1 avg: {}".format(
            val[0], val[1], val[2], val[3], val[4]))


# Callback für F1-Werte über Epochen, nicht über Batches
class Metrics(Callback):
    def __init__(self, name, i):
        self.name = name
        self.i = i

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.epoch = 0
        self.ind = 0

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" — val_f1: % f — val_precision: % f — val_recall % f" %
              (_val_f1, _val_precision, _val_recall))
        print("confusion_matrix\n", confusion_matrix(val_targ, val_predict))
        # Auf höchsten F1-Wert überprüfen und ggf. Datei auslagern
        if _val_f1 >= max(self.val_f1s):
            self.model.save("model_{}_{}.h5".format(self.name, self.i))
            self.ind = self.epoch
            print("model_{}_{} ausgelagert".format(self.name, str(self.i)))
        self.epoch += 1
        return


if __name__ == '__main__':
    print("Upsampling files...")
    upsample_lines()
    print("Wrote ./data/reddit_upsample.tsv")
