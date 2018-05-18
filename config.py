img_rows, img_cols = 320, 320
img_rows_half, img_cols_half = 160, 160
channel = 4
batch_size = 16
epochs = 1000
patience = 50
num_samples = 43100
num_train_samples = 34480
# num_samples - num_train_samples
num_valid_samples = 8620
unknown = 128
# bgr     others      car         motorbicycle bicycle         person        truck         bus              tricycle
colors = [[0, 0, 0], [255, 0, 0], [17, 7, 65], [22, 118, 174], [0, 0, 255], [0, 220, 219], [40, 170, 241], [153, 153, 189]]
