import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, LSTM

input_shape = (5, 17 * 2)
num_classes = 2
source_path = './source'
tag = [[0], [1]]
tag_class = {'timber': 0, 'walk': 1}
output_tag = {0: 'timber', 1: 'walk'}


# Define a simple sequential model
# return_sequences一般在最后一层为False
def create_model():
    model = tf.keras.Sequential([
        LSTM(300, return_sequences=True, input_shape=input_shape),
        Dropout(0.1),
        LSTM(100),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, clipnorm=1),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'], )
    return model


def change_shape(arr=None):
    if arr is None:
        arr = []
    arr_changed = np.ones((5, 17 * 2))
    for i in range(0, 17):
        for j in range(0, 2):
            arr_changed[0][i * 2 + j] = arr[0][i][j]
            arr_changed[1][i * 2 + j] = arr[1][i][j]
            arr_changed[2][i * 2 + j] = arr[2][i][j]
            arr_changed[3][i * 2 + j] = arr[3][i][j]
            arr_changed[4][i * 2 + j] = arr[4][i][j]
    return arr_changed

# Create a basic model instance
model_lstm = create_model()

checkpoint_save_path = "./checkpoint/lstm.ckpt"
model_lstm.load_weights(checkpoint_save_path)

data_lstm = np.load('./source/DATA_2_origin/2/walk/1/0.npy')
print(data_lstm.shape)
data_reshape = change_shape(data_lstm)
data_reshape = np.reshape(data_reshape, (1, 5, 17 * 2))
result_lstm = model_lstm.predict([data_reshape])
predict_tag = tf.argmax(result_lstm, axis=1)

predict_tag = int(predict_tag)
print(output_tag[predict_tag])
