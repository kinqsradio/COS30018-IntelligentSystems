from initialized import *
import matplotlib.pyplot as plt
from models import create_dynamic_model

from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler
  
def intialize_model(input_shape, layer_configs):
    model = create_dynamic_model(input_shape, layer_configs)
    model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])
    model.summary()
    
    return model
    
def train_model(model, x_train, y_train, batch_size, epochs):
    model_save_path = os.path.join(SKELETON_DIR, 'best_model.pt')
    csv_log_file = os.path.join(SKELETON_DIR, 'training_log.csv')
    
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    checkpoint_acc = ModelCheckpoint(os.path.join(SKELETON_DIR, 'best_model_acc.h5'), monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    tensorboard = TensorBoard(log_dir=os.path.join(SKELETON_DIR, 'logs'), histogram_freq=1, write_graph=True, write_images=True)
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
    return history


# Function to plot training metrics
def plot_metrics(history):
    plt.figure(figsize=(16, 12))
    
    # Training vs. Validation Loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss Evolution')
    
    # Training vs. Validation Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy Evolution')
    
    # Learning rate (if available)
    if 'lr' in history.history:
        plt.subplot(2, 2, 3)
        plt.plot(history.history['lr'], label='Learning Rate')
        plt.legend()
        plt.title('Learning Rate Evolution')
    
    # Additional metrics (if any are available, beyond the ones already plotted)
    available_metrics = set(history.history.keys()) - {'loss', 'val_loss', 'accuracy', 'val_accuracy', 'lr'}
    for idx, metric in enumerate(available_metrics, start=4):
        plt.subplot(2, 3, idx)
        plt.plot(history.history[metric], label=f'Train {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.legend()
        plt.title(f'{metric} Evolution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SKELETON_DIR, 'training_metrics.png'))
    plt.show()
  