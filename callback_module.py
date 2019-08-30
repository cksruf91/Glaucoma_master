import pandas as pd
import numpy as np
from slacker import Slacker

from keras.callbacks import Callback
import keras.backend as K

from sklearn import metrics
from sklearn.metrics import confusion_matrix, mean_squared_error
from utils.util import slack_message, confusion_matrix_report
    
class SlackMessage(Callback):
    def __init__(self , slack_token,monitor_name):
        self.slack_token = slack_token
        self.monitor_name = monitor_name

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self,epoch, logs={}):
        moniter = 'mean_squared_error' if self.monitor_name == 'mse' else self.monitor_name
        #self.logs.append(logs)
        self.losses = logs.get('loss')
        self.monitor = logs.get(moniter)
        self.val_losses = logs.get('val_loss')
        self.val_monitor = logs.get('val_'+self.monitor_name)
        message = f"epoch : {epoch+1} | loss : {self.losses:.5f}, {moniter} : {self.monitor:.5f}, val_loss : {self.val_losses:.5f}, {'val_'+self.monitor_name} : {self.val_monitor:.5f}"
        slack_message('#glaucoma', message, self.slack_token)

class IntervalEvaluation(Callback):
    """ 매 epoch 마다 loss 와 지정된 monitoring method 기록
    
    Arguments:
        Callback {class} -- keras api
    """
    def __init__(self , val_generator,loss_func,monitor_name='auc'):
        self.val_generator = val_generator
        self.monitor_name = monitor_name
        self.loss_func = loss_func
        self.y_true = val_generator.get_label()

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self,epoch, logs={}):
        y_pred = self.model.predict_generator(self.val_generator,
                                            steps=None,
                                            max_queue_size=10,
                                            workers=2,
                                            use_multiprocessing=False,
                                            verbose=0
                                             )
#         print(self.y_true.shape)
#         print(y_pred.shape)
        if self.loss_func == 'binary_crossentropy':
            loss = binary_cross_entropy(self.y_true, y_pred)
            y_true = self.y_true
        elif self.loss_func == 'categorical_crossentropy':
            loss = categorical_crossentropy(self.y_true, y_pred)
            y_true = np.argmax(self.y_true, axis = -1)
            y_pred = np.argmax(y_pred, axis = -1)
            
        elif self.loss_func == 'hinge':
            loss = hinge_function(self.y_true, y_pred)
        else :
            raise ValueError("unavailable loss function")
        
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        if self.monitor_name == 'auc':
            monitor = auc_function(y_true, y_pred)
        elif self.monitor_name == 'mse':
            monitor = mean_squared_error(y_true, y_pred)
        else :
            raise ValueError("unavailable monitor method")
        
        logs['val_loss']  = loss
        logs[f'val_{self.monitor_name}']  = monitor
        
        print(f" - val_loss : {loss:0.5f}  - val_{self.monitor_name} : {monitor:0.5f}")
        conf_mat, sensitivity, specificity = confusion_matrix_report(y_true, y_pred,0.5)
        print(conf_mat)
        print(sensitivity , specificity)
        
        
        
class HistoryCheckpoint(Callback):
    """ 매 epoch 마다 loss 와 지정된 monitoring method 기록
    
    Arguments:
        Callback {class} -- keras api
    """
    def __init__(self , path, monitor_name):
        self.path = path
        self.monitor_name = monitor_name

    def on_train_begin(self, logs={}):
        self.logs = []

    def on_epoch_end(self,epoch, logs={}):
        moniter = 'mean_squared_error' if self.monitor_name == 'mse' else self.monitor_name
        self.logs.append(logs)
        self.losses = logs.get('loss')
        self.monitor = logs.get(moniter)
        self.val_losses = logs.get('val_loss')
        self.val_monitor = logs.get('val_'+self.monitor_name)
        
        if epoch == 0:
            with open(self.path , 'w') as f:
                line = f"epoch,loss,{self.monitor_name},val_loss,{'val_'+self.monitor_name}" 
                f.write(line+'\n')
        with open(self.path , 'a') as f:
            line = f"{epoch+1},{self.losses},{self.monitor},{self.val_losses},{self.val_monitor}" 
            f.write(line+'\n')
    

def binary_cross_entropy(y_true, y_pred):
    loss= []
    if isinstance(y_true,np.ndarray):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
    
    for t,p in zip(y_true,y_pred):
        p = np.clip(p, 0.+K.epsilon(), 1.-K.epsilon())
        cross_entropy = np.float( t*np.log(p) + (1-t)*np.log((1-(p+K.epsilon()))) )
        loss.append(cross_entropy)
    return -np.mean(loss)

def auc_function(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    return metrics.auc(fpr, tpr) 

def hinge_function(y_true, y_pred):
    y_true_ = np.copy(y_true)
    y_true_[y_true_==0] = -1
    zeros = np.zeros(y_true_.shape)
    hinge_loss=np.maximum(zeros,(1- y_true_*y_pred))
    return np.mean(hinge_loss)

def categorical_crossentropy(y_true, y_pred):
    losses = []
    for _ in range(y_true.shape[0]):
        cliped = np.clip(y_pred, 0+ K.epsilon(), 1-K.epsilon())
        losses.append(np.mean(-np.sum(y_true*np.log(cliped), axis = -1)) )
    return np.mean(losses)