import time
import sys
sys.path.append('..')

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from slacker import Slacker
from config import *


def history_graph(hist):
    fig = plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('acc(%)')
    ax_acc = fig.add_subplot(111)
    line1 = ax_acc.plot(hist['epoch'], hist['acc'], label='acc', color='#0613a3')
    line2 = ax_acc.plot(hist['epoch'], hist['val_acc'], label='val_acc', color='#7311d6')
    ax_loss = ax_acc.twinx()
    line3 = ax_loss.plot(hist['epoch'], hist['loss'], label='loss', color='#a52121')
    line4 = ax_loss.plot(hist['epoch'], hist['val_loss'], label='val_loss', color='#c48a03')
    plt.ylabel('loss')

    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels, fancybox=True, bbox_to_anchor=(1.35, 1.05))
    plt.show()
    return None


def print_confusion_matrix(y_true, y_pred, cut_off=0.5):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred > cut_off).ravel()
    print(f"{' ':5}{1:5},{0:5}\n{1:5}{tp:5},{fp:5}\n{0:5}{fn:5},{tn:5}")
    return None


def slack_message(channel, message):
    slack = Slacker(MY_SLACK_TOKEN)
    slack.chat.post_message(channel, message)


def learning_rate_schedule(epoch_, lr):
    if epoch_ > 80:
        lr *= 0.5e-3
    elif epoch_ > 35:
        lr *= 1e-3
    elif epoch_ > 30:
        lr *= 1e-2
    elif epoch_ > 15:
        lr *= 1e-1
    return lr


def train_progressbar(iteration, total, epoch=0, epochs=0, loss=.0, acc=.0, decimals=1, barLength=100):
    formatStr = '{0:.' + str(decimals) + 'f}'
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '>' * filledLength + ' ' * (barLength - filledLength)
    sys.stdout.write(
        '\r epoch: {}/{} [{}] {} % - loss : {:5.5f}, - acc : {:5.5f}'.format(epoch, epochs, bar, percent, loss, acc)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


class Progress:
    def __init__(self):
        self.count = 1

    def add_count(self):
        self.count += 1

    def print_progress(self, batch_size, total, j):
        dot_num = int(batch_size * self.count / total * 100)
        dot = '>' * dot_num
        empty = '_' * (100 - dot_num)
        sys.stdout.write('\r [{dot}{empty}] {j} Done'.format(dot=dot, empty=empty, j=j))
        self.add_count()


if __name__ == '__main__':
    #     slack_message('#resnet_project', 'hi hello')

    for i in range(0, 100):
        train_progressbar(iteration=i, total=100, epoch=1, epochs=100, loss=1.1, acc=0.8, decimals=1, barLength=50)
        time.sleep(0.05)
