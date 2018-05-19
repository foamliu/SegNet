import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import migrate
from config import patience, batch_size, epochs, num_train_samples, num_valid_samples
from data_generator import train_gen, valid_gen
from model import build_encoder_decoder
from utils import get_available_cpus

if __name__ == '__main__':
    checkpoint_models_path = 'models/'

    # Callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    model_names = checkpoint_models_path + 'model.{epoch:02d}-{val_acc:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1, save_best_only=True)
    early_stop = EarlyStopping('val_acc', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_acc', factor=0.1, patience=int(patience / 4), verbose=1)


    class MyCbk(keras.callbacks.Callback):
        def __init__(self, model):
            keras.callbacks.Callback.__init__(self)
            self.model_to_save = model

        def on_epoch_end(self, epoch, logs=None):
            fmt = checkpoint_models_path + 'model.%02d-%.4f.hdf5'
            self.model_to_save.save(fmt % (epoch, logs['val_acc']))


    # num_gpu = len(get_available_gpus())
    # if num_gpu >= 2:
    #     with tf.device("/cpu:0"):
    #         model = build_encoder_decoder()
    #         migrate.migrate_model(model)
    #
    #     new_model = multi_gpu_model(model, gpus=num_gpu)
    #     # rewrite the callback: saving through the original model and not the multi-gpu model.
    #     model_checkpoint = MyCbk(model)
    # else:
    model = build_encoder_decoder()
    migrate.migrate_model(model)

    model.compile(optimizer='nadam', loss='sparse_softmax_cross_entropy_with_logits', metrics=['accuracy'])

    print(model.summary())

    # Final callbacks
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    # Start Fine-tuning
    model.fit_generator(train_gen(),
                        steps_per_epoch=num_train_samples // batch_size,
                        validation_data=valid_gen(),
                        validation_steps=num_valid_samples // batch_size,
                        shuffle=True,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        use_multiprocessing=True,
                        workers=int(round(get_available_cpus() / 2))
                        )
