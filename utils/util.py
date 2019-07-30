import numpy as np
import sys
sys.path.append("..")
import time
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from slacker import Slacker
from config import *


def history_graph(hist,metics): 
    fig = plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('acc(%)')
    ax_acc = fig.add_subplot(111)
    line1 = ax_acc.plot(hist['epoch'], hist[metics], label=metics, color='#0613a3') 
    line2 = ax_acc.plot(hist['epoch'], hist['val_'+metics], label='val_'+metics ,color='#7311d6') 
    ax_loss = ax_acc.twinx()
    line3 = ax_loss.plot(hist['epoch'], hist['loss'], label='loss', color='#a52121')
    line4 = ax_loss.plot(hist['epoch'], hist['val_loss'], label='val_loss', color='#c48a03')
    plt.ylabel('loss')

    lines = line1+line2+line3+line4
    labels = [l.get_label() for l in lines]
    plt.legend(lines,labels, fancybox=True, bbox_to_anchor=(1.35, 1.05))
    plt.show()
    return None

def print_confusion_matrix(y_true,y_pred,cut_off = None):
    if cut_off is None:
        cut_off = np.quantile(y_pred,0.9)
#     print(f"cut_off : {cut_off:5}")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred>cut_off).ravel()
    print(f"{cut_off:3.1f}{1:7},{0:7}\n{1:7}{tp:7},{fp:7}\n{0:7}{fn:7},{tn:7}")
    return None

def slack_message(chennel, message):
    slack = Slacker(MY_SLACK_TOKEN)
    slack.chat.post_message(chennel, message)

def learning_rate_schedule(epoch_, lr):
    if epoch_ > 55:
        lr *= 1e-4
    elif epoch_ > 45:
        lr *= 0.5e-3
    elif epoch_ > 35:
        lr *= 1e-3
    elif epoch_ > 25:
        lr *= 1e-2
    elif epoch_ > 15:
        lr *= 1e-1
    return lr


def train_progressbar(iteration, total, epoch = '',epochs = '', loss = '', monitor='' ,acc = '', decimals = 1, barLength = 100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '>' * filledLength + ' ' * (barLength - filledLength)
    sys.stdout.write("\r epoch: {}/{} [{}] {} % - loss : {:5.5f}, - {} : {:5.5f}".format(
        epoch, epochs, bar, percent, loss, monitor, acc)),
#     if iteration == total:
#         sys.stdout.write('\n')
    sys.stdout.flush()


class progress():
    def __init__(self):
        self.count = 1
    def add_count(self):
        self.count += 1
    
    def print_progress(self,batch_size,total,i):
        dot_num = int((batch_size*self.count)/total*100)
        dot = '>'*dot_num
        empty = '_'*(100-dot_num)
        sys.stdout.write(f'\r [{dot}{empty}] {i} Done')
        if i == total:
            sys.stdout.write('\n')
        self.add_count()
        

if __name__ == "__main__":
#     slack_message('#resnet_project', 'hi hello')
    
    for i in range(0, 100):
        train_progressbar(iteration =i , total =100, epoch = 1 ,epochs = 100, loss = 1.1, acc = 0.8 , decimals = 1, barLength = 50)
        time.sleep(0.05)
        
        
        