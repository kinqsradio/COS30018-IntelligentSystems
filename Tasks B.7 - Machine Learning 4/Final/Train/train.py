import os
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import Trainer, TrainingArguments
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler

'''LSTM'''
def train_lstm(model, x_train, y_train, batch_size, epochs):
    save_path = 'model/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_save_path = os.path.join(save_path, 'best_model.pt')
    csv_log_file = os.path.join(save_path, 'training_log.csv')
    
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    checkpoint_acc = ModelCheckpoint(os.path.join(save_path, 'best_model_acc.h5'), monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    tensorboard = TensorBoard(log_dir=os.path.join(save_path, 'logs'), histogram_freq=1, write_graph=True, write_images=True)
    csv_logger = CSVLogger(csv_log_file, append=True)
    
    # Increase patience
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=15, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)

    # Learning rate scheduler
    def lr_schedule(epoch):
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 5e-4
        elif epoch > 120:
            lr *= 1e-3
        elif epoch > 80:
            lr *= 5e-3
        print('Learning rate: ', lr)
        return lr

    lr_scheduler = LearningRateScheduler(lr_schedule)
    callbacks_list = [checkpoint, checkpoint_acc, tensorboard, csv_logger, reduce_lr, early_stopping, lr_scheduler]


    # Training Execution
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.2, callbacks=callbacks_list)
    plot_metrics(history, model_name='lstm')
    return history

# Testing prediction on train model
def lstm_predict_test(model, train_target_scaler, x_test, y_test):    
    # Predictions on test data
    predictions = model.predict(x_test)
    predictions = train_target_scaler.inverse_transform(predictions)
        
    return predictions  
  
'''FINBERT'''
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    return {'accuracy':  accuracy_score(predictions, labels)}
       
            
def trainer_finbert(model, tokenizer, train_datasets, val_datasets, epochs, save_path='finbert'):
    
    args = TrainingArguments(
        output_dir = 'temp/',
        evaluation_strategy = 'epoch',
        save_strategy = 'epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy')  
    
    
    trainer = Trainer(
        model = model,
        args = args,
        train_dataset= train_datasets,
        eval_dataset= val_datasets,
        compute_metrics = compute_metrics)
    
    
    
    trainer.train()
    
    # Capture all metrics dynamically from trainer's log history
    history = {}
    for entry in trainer.state.log_history:
        for key, value in entry.items():
            # Convert naming from 'eval_*' to 'val_*' 
            formatted_key = key.replace("eval_", "val_")
            if formatted_key not in history:
                history[formatted_key] = []
            history[formatted_key].append(value)
    
    plot_metrics(history, model_name='finbert')
    
    save_model(trainer, tokenizer, path=save_path)
    return trainer

def save_model(trainer, tokenizer, path):
    trainer.save_model(path)
    tokenizer.save_pretrained(path)
    
    
    
'''Plot Metrics'''
# Function to plot training metrics
def plot_metrics(history, model_name):
    save_path = 'training_history/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Adapt for Keras' History object
    if not isinstance(history, dict):
        history_data = history.history
    else:
        history_data = history

    loss = history_data.get('loss', [])
    val_loss = history_data.get('val_loss', [])
    accuracy = history_data.get('accuracy', [])
    val_accuracy = history_data.get('val_accuracy', [])
    lr = history_data.get('lr', [])

    metrics_to_plot = ['loss', 'accuracy', 'lr']
    metrics_to_plot += [metric for metric in history_data.keys() if metric not in {'loss', 'val_loss', 'accuracy', 'val_accuracy', 'lr'}]
    
    total_metrics = len(metrics_to_plot)
    num_columns = int(math.sqrt(total_metrics))  # We'll have a roughly square layout
    num_rows = math.ceil(total_metrics / num_columns)

    plt.figure(figsize=(5*num_columns, 4*num_rows))

    for idx, metric in enumerate(metrics_to_plot, start=1):
        plt.subplot(num_rows, num_columns, idx)
        
        if metric == "loss":
            plt.plot(loss, label='Train Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.title('Loss Evolution')
        elif metric == "accuracy":
            plt.plot(accuracy, label='Train Accuracy')
            plt.plot(val_accuracy, label='Validation Accuracy')
            plt.title('Accuracy Evolution')
        elif metric == "lr":
            plt.plot(lr, label='Learning Rate')
            plt.title('Learning Rate Evolution')
        else:
            train_metric = history_data[metric]
            val_metric_key = f'val_{metric}'
            val_metric = history_data.get(val_metric_key, [])
            plt.plot(train_metric, label=f'Train {metric}')
            if val_metric:
                plt.plot(val_metric, label=f'Validation {metric}')
            plt.title(f'{metric} Evolution')

        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{model_name}_training_metrics.png'))
    plt.show()
