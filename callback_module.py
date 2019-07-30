import pandas as pd
import numpy as np

from keras.callbacks import Callback

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from utils.util import train_progressbar, slack_message, learning_rate_schedule, print_confusion_matrix

def binary_cross_entropy(y_true, y_pred):
    loss= []
    for t,p in zip(y_true,y_pred):
        cross_entropy = np.float( t*np.log(p) + (1-t)*np.log((1-p)) )
        loss.append(cross_entropy)
    return -np.mean(loss)

def auc_function(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    return metrics.auc(fpr, tpr) 

class IntervalEvaluation(Callback):
    """ 매 epoch 마다 loss 와 지정된 monitoring method 기록
    
    Arguments:
        Callback {class} -- keras api
    """
    def __init__(self , val_generator,monitor_name='auc'):
        self.val_generator = val_generator
        self.monitor_name = monitor_name

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self,epoch, logs={}):
        y_pred = self.model.predict_generator(self.val_generator,
                                            steps=None,
                                            max_queue_size=10,
                                            workers=1,
                                            use_multiprocessing=False,
                                            verbose=0
                                             )
        y_true = self.val_generator.get_label()
        loss = binary_cross_entropy(y_true, y_pred)
        if self.monitor_name == 'auc':
            monitor = auc_function(y_true, y_pred)
        else :
            raise ValueError("invalied monitor")
        
        logs['val_loss']  = loss
        logs[f'val_{self.monitor_name}']  = monitor
        
        print(f" - val_loss : {loss}  - val_{self.monitor_name} : {monitor}")
        print_confusion_matrix(y_true, y_pred,0.5)
        
        
class HistoryCheckpoint(Callback):
    """ 매 epoch 마다 loss 와 지정된 monitoring method 기록
    
    Arguments:
        Callback {class} -- keras api
    """
    def __init__(self , path, metrics_name):
        self.path = path
        self.metrics_name = metrics_name

    def on_train_begin(self, logs={}):
        self.losses = []
        self.metrics = []
        self.val_losses = []
        self.val_metrics = []
        self.logs = []

    def on_epoch_end(self,epoch, logs={}):
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.metrics.append(logs.get(str(self.metrics_name)))
        self.val_losses.append(logs.get('val_loss'))
        self.val_metrics.append(logs.get(str('val_'+self.metrics_name)))

        hist = pd.DataFrame({
            'losses' : self.losses
            ,self.metrics_name : self.metrics
            ,'val_losses' : self.val_losses
            ,'val_'+self.metrics_name : self.val_metrics
            })
        
        hist.to_csv(self.path)