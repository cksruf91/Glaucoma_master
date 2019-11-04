import numpy as np
import sys
sys.path.append("..")
import time
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from slacker import Slacker
from config import *

class printer():
    def __init__(self, obj, total, bar_length, prefix):
        self.idx = 0
        self.obj = obj
        self.total = total
        self.bar_length = bar_length
        self.prefix = prefix

    def progressbar(self, total, i, bar_length, prefix):
        dot_num = int(i/total*bar_length)
        dot = '>'*dot_num
        empty = '_'*(bar_length-dot_num)
        sys.stdout.write(f'\r {prefix} [{dot}{empty}] {i/total*100:3.2f} % Done')
        if i == total:
            sys.stdout.write('\n')
    
    def __next__(self):
        self.progressbar(self.total, self.idx, self.bar_length, self.prefix)
        self.idx += 1
        return next(self.obj)

class pbar():
    def __init__(self, iterable, bar_length=50, prefix=""):
        self.iterable = iterable
        self.total = len(self.iterable)
        self.bar_length = bar_length
        self.prefix = prefix
    
    def __iter__(self):
        iterobj = iter(self.iterable)
        return printer(iterobj, self.total, self.bar_length, self.prefix)
    

def last_cheackpoint(checkpoint_dir):
    checkpoints = [i for i in os.listdir(checkpoint_dir) if 'checkpoint-' in i ]
    checkpoints.sort(key = lambda s : os.path.getmtime(os.path.join(checkpoint_dir, s)))
    return os.path.join(checkpoint_dir,checkpoints[-1])

def get_config(obj):
    if hasattr(obj,"__class__")==False:
        raise ValueError(f"{obj} is not class")
    return {obj.__class__.__name__ : obj.get_config()}

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

def confusion_matrix_report(y_true,y_pred,cut_off = None):
    if cut_off is None:
        cut_off = np.quantile(y_pred,0.9)
#     print(f"cut_off : {cut_off:5}")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred>cut_off).ravel()
    conf_mat = f"{cut_off:0.5f}{1:7},{0:7}\n{1:7}{tp:7},{fp:7}\n{0:7}{fn:7},{tn:7}"
    sensitivity = f"Sensitivity : {tp/(tp+fn):1.5}"
    specificity = f"Specificity : {tn/(tn+fp):1.5}"
    return conf_mat, sensitivity, specificity

def slack_message(chennel, message, token):
    slack = Slacker(token)
    slack.chat.post_message(chennel, message)


def visualize_anomaly(error_df, threshold = None):
    """anomaly graph 출력
    
    Arguments:
        error_df {pandas DataFrame} -- Class, y_pred column이 포함된 DataFrame
                clss : label
                y_pred : 예측된 스코어
    
    Keyword Arguments:
        threshold {int} -- 그래프에 출력될 cut off값, 미지정시 상위 0.5% 자동  (default: {None})
    """
    if threshold is None:
        threshold = error_df[error_df['Class'] == 1].y_pred.quantile(q = 0.5) # 95 % higher
        print('Generated threshold : {}'.format(threshold))
        
    fig, ax = plt.subplots(figsize = (10,6))

    for name, group in error_df.groupby('Class'):
        ax.plot(group.index, group['y_pred'], marker = 'o', linestyle = '', alpha = 0.6, 
                label = "Fraud" if name == 1 else "Normal",
                color = 'r' if name == 1 else 'royalblue')

    ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], 
              colors = 'r', zorder = 100, label = 'Threshold')
    ax.legend()
        

if __name__ == "__main__":
    slack_message('#resnet_project', 'hi hello')
#     for i in range(0, 100):
#         train_progressbar(iteration =i , total =100, epoch = 1 ,epochs = 100, loss = 1.1, acc = 0.8 , decimals = 1, barLength = 50)
#         time.sleep(0.05)
        
        
        