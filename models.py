import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
# tf.enable_eager_execution()
import numpy as np
from numba import jit


class DenseLayer(Model):
    def __init__(self, n_units, drop_rate=.5, activation=tf.nn.elu):
        super(DenseLayer, self).__init__()
        self.dense = layers.Dense(n_units,
                                  activation=activation,
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=keras.regularizers.l2(0.01))

        self.batchnorm = layers.BatchNormalization()
        self.drop = layers.Dropout(rate=drop_rate)

    def call(self, inputs, training=False):
        layer = self.dense(inputs)
        layer = self.batchnorm(layer)
        layer = tf.nn.elu(layer)
        layer = self.drop(layer)

        return layer


class LSTMLayer(Model):
    def __init__(self, n_units, drop_rate=.3, return_sequences=True):
        super(LSTMLayer, self).__init__()
        self.LSTM = layers.CuDNNLSTM(n_units,
                                     return_sequences=return_sequences,
                                     kernel_initializer='he_normal',
                                     kernel_regularizer=keras.regularizers.l2(0.01))
        self.batchnorm = layers.BatchNormalization()
        self.drop = layers.Dropout(rate=drop_rate)
        # self.drop = layers.TimeDistributed(keras.layers.Dropout(rate = drop_rate))

    def call(self, inputs, training=False):
        layer = self.LSTM(inputs)
        layer = self.batchnorm(layer)
        layer = self.drop(layer)

        return layer


def ConvLayer(inputs, drop_rate=0.8, filters=8, kernel_size=(1, 100), strides=(1, 10), padding='SAME'):
    conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                         kernel_initializer='he_normal',
                         kernel_regularizer=keras.regularizers.l2(0.01))(inputs)

    batchnorm = layers.BatchNormalization()(conv)
    activation = tf.nn.relu6(batchnorm)
    dropout = layers.Dropout(rate=drop_rate)(activation)

    return dropout


"""
class ConvLayer(tf.keras.Model):
    def __init__(self,drop_rate = 0.3, filters = 8, kernel_size = (1,100), strides = (1,10), padding = 'SAME', activation = tf.nn.relu6):
        super(ConvLayer, self).__init__()
        self.conv = keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides,padding = padding,
                                        
                                        kernel_initializer='he_normal',
                                        kernel_regularizer=keras.regularizers.l2(0.01))
        
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.dropout = keras.layers.Dropout(rate = drop_rate)
    def call(self,inputs, training = False):
        layer = self.conv(inputs)
        layer = self.batchnorm(layer)
        layer = tf.nn.relu6(layer)
        layer = self.dropout(layer)
        return layer
"""


def DLNN(time_step, n_input, dnn_units, lstm_uits):
    with tf.device('/gpu:0'):
        inputs = keras.Input(shape=(time_step, n_input))
        dense1 = DenseLayer(dnn_units[0])(inputs)
    with tf.device('/gpu:1'):
        dense2 = DenseLayer(dnn_units[1])(dense1)
        dense3 = DenseLayer(dnn_units[2])(dense2)
    with tf.device('/gpu:2'):
        lstm1 = LSTMLayer(lstm_uits)(dense3)
        lstm2 = LSTMLayer(lstm_uits)(lstm1)
        lstm3 = LSTMLayer(lstm_uits)(lstm2)
    with tf.device('/gpu:3'):
        stacked_rnn_outputs = tf.reshape(lstm3, [-1, lstm_uits])
        stacked_outputs = keras.layers.Dense(units=1)(stacked_rnn_outputs)
        logits = tf.reshape(stacked_outputs, [-1, time_step, 1])

    return Model(inputs=inputs, outputs=logits)


def CLNN(time_step, n_input, lstm_uits):
    with tf.device('/gpu:0'):
        inputs = keras.Input(shape=(time_step, n_input, 1))
        conv1 = ConvLayer(inputs)
        # conv1 = ConvLayer()(inputs)
    with tf.device('/gpu:1'):
        conv2 = ConvLayer(conv1)
        CNN_output = tf.reshape(conv2, (-1, conv2.shape[1], conv2.shape[2] * conv2.shape[3]))

    with tf.device('/gpu:2'):
        lstm1 = LSTMLayer(lstm_uits)(CNN_output)
        lstm2 = LSTMLayer(lstm_uits)(lstm1)
        lstm3 = LSTMLayer(lstm_uits)(lstm2)

        stacked_rnn_outputs = tf.reshape(lstm3, [-1, lstm_uits])
        stacked_outputs = keras.layers.Dense(units=1)(stacked_rnn_outputs)
        logits = tf.reshape(stacked_outputs, [-1, time_step, 1])

    return Model(inputs=inputs, outputs=logits)


class AttentionLayer(Model):
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        layer1 = self.W1(inputs[0])
        layer2 = self.W2(inputs[1])
        score = tf.matmul(layer1, tf.transpose(layer2, [0, 2, 1]))
        dim = score.shape[2]
        score = score / np.sqrt(int(dim))
        distribution = tf.nn.softmax(score)
        att = tf.matmul(distribution, inputs[2])
        return att


def LSTM(n_timestep, n_inputs, n_units, regularizers_alpha=0.01, drop_rate=0.5, gpu=0):
    with tf.device('/gpu:' + str(gpu)):
        inputs = keras.Input(shape=(n_timestep, n_inputs))
        # dropout = layers.Dropout(rate = drop_rate)(inputs)
        lstm1 = layers.LSTM(n_units,
                            return_sequences=True,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(inputs)
        batchnorm1 = layers.BatchNormalization()(lstm1)

        lstm2 = layers.LSTM(n_units,
                            return_sequences=True,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm1)
        batchnorm2 = layers.BatchNormalization()(lstm2)

        lstm3 = layers.LSTM(n_units,
                            return_sequences=True,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm2)
        batchnorm3 = layers.BatchNormalization()(lstm3)

        stacked_rnn_outputs = tf.reshape(batchnorm3, [-1, n_units])
        stacked_outputs = keras.layers.Dense(units=1)(stacked_rnn_outputs)
        logits = tf.reshape(stacked_outputs, [-1, n_timestep, 1])

        return Model(inputs=inputs, outputs=logits)


def ResLSTM1(n_timestep, n_inputs, n_units, regularizers_alpha=0.01, drop_rate=0.5, gpu=0):
    with tf.device('/gpu:' + str(gpu)):
        inputs = keras.Input(shape=(n_timestep, n_inputs))
        dropout = layers.Dropout(rate=drop_rate)(inputs)
        lstm1 = layers.CuDNNLSTM(n_inputs,
                                 return_sequences=True,
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(dropout)
        add1 = layers.add([lstm1, inputs])
        batchnorm1 = layers.BatchNormalization()(add1)

        lstm2 = layers.CuDNNLSTM(n_units,
                                 return_sequences=True,
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm1)
        batchnorm2 = layers.BatchNormalization()(lstm2)

        lstm3 = layers.CuDNNLSTM(n_units,
                                 return_sequences=True,
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm2)
        add2 = layers.add([lstm3, batchnorm2])
        batchnorm3 = layers.BatchNormalization()(add2)

        stacked_rnn_outputs = tf.reshape(add2, [-1, n_units])
        stacked_outputs = keras.layers.Dense(units=1)(stacked_rnn_outputs)
        logits = tf.reshape(stacked_outputs, [-1, n_timestep, 1])

        return Model(inputs=inputs, outputs=logits)


def ResLSTM2(n_timestep, n_inputs, n_units, regularizers_alpha=0.01, drop_rate=0.5, gpu=0):
    with tf.device('/gpu:' + str(gpu)):
        inputs = keras.Input(shape=(n_timestep, n_inputs))
        dropout = layers.Dropout(rate=drop_rate)(inputs)
        lstm1 = layers.LSTM(n_units,
                            return_sequences=True,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(dropout)
        batchnorm1 = layers.BatchNormalization()(lstm1)

        lstm2 = layers.LSTM(n_units,
                            return_sequences=True,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm1)

        add1 = layers.add([lstm2, batchnorm1])

        batchnorm2 = layers.BatchNormalization()(add1)

        lstm3 = layers.LSTM(n_inputs,
                            return_sequences=True,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm2)

        add2 = layers.add([lstm3, inputs])

        batchnorm3 = layers.BatchNormalization()(add2)

        stacked_rnn_outputs = tf.reshape(batchnorm3, [-1, batchnorm3.shape[2]])
        stacked_outputs = keras.layers.Dense(units=1)(stacked_rnn_outputs)
        logits = tf.reshape(stacked_outputs, [-1, n_timestep, 1])

        return Model(inputs=inputs, outputs=logits)


def ResLSTM2noise(n_timestep, n_inputs, n_units, regularizers_alpha=0.01, drop_rate=0.5, gpu=0):
    with tf.device('/gpu:' + str(gpu)):
        inputs = keras.Input(shape=(n_timestep, n_inputs))
        dropout = layers.Dropout(rate=drop_rate)(inputs)
        lstm1 = layers.LSTM(n_units,
                            return_sequences=True,

                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(dropout)

        batchnorm1 = layers.BatchNormalization()(lstm1)

        lstm2 = layers.LSTM(n_units,
                            return_sequences=True,

                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm1)

        add1 = layers.add([lstm2, batchnorm1])

        batchnorm2 = layers.BatchNormalization()(add1)

        lstm3 = layers.LSTM(n_inputs,
                            return_sequences=True,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm2)

        batchnorm3 = layers.BatchNormalization()(lstm3)

        stacked_rnn_outputs = tf.keras.layers.Reshape((-1, batchnorm3.shape[2]))(batchnorm3)
        stacked_outputs = keras.layers.Dense(units=1)(stacked_rnn_outputs)
        logits = tf.keras.layers.Reshape((n_timestep, 1), name='logits')(stacked_outputs)

        sigma_init = tf.random_normal_initializer()
        mu_init = tf.random_normal_initializer()

        sigma = abs(tf.Variable(sigma_init(shape=(1, n_timestep, 1)), dtype='float32', trainable=True, name='sigma'))

        print('sigma size :', sigma.shape)
        print('logits shape:', logits.shape)
        print('tf.random.normal(shape=tf.shape(logits) :', tf.random.normal(shape=tf.shape(logits)).shape)
        random_noise = tf.random.normal(shape=tf.shape(logits))
        noise = tf.math.multiply(random_noise, sigma)
        print('noise shape :', noise.shape)
        logits_with_noise = layers.add([logits, noise])

        return Model(inputs=inputs, outputs=logits_with_noise)


def ResLSTM3(n_timestep, n_inputs, n_units, regularizers_alpha=0.01, drop_rate=0.5, gpu=0):
    with tf.device('/gpu:' + str(gpu)):
        inputs = keras.Input(shape=(n_timestep, n_inputs))
        dropout = layers.Dropout(rate=drop_rate)(inputs)
        lstm1 = layers.CuDNNLSTM(n_units,
                                 return_sequences=True,
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(dropout)
        batchnorm1 = layers.BatchNormalization()(lstm1)

        lstm2 = layers.CuDNNLSTM(n_units,
                                 return_sequences=True,
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm1)

        add1 = layers.add([lstm2, batchnorm1])

        batchnorm2 = layers.BatchNormalization()(add1)

        lstm3 = layers.CuDNNLSTM(n_units,
                                 return_sequences=True,
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm2)
        add2 = layers.add([lstm3, batchnorm2])

        batchnorm3 = layers.BatchNormalization()(add2)

        lstm4 = layers.CuDNNLSTM(n_units,
                                 return_sequences=True,
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm3)
        add3 = layers.add([lstm4, batchnorm3])

        batchnorm4 = layers.BatchNormalization()(add3)

        lstm5 = layers.CuDNNLSTM(n_inputs,
                                 return_sequences=True,
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm4)

        add4 = layers.add([lstm5, inputs])

        batchnorm5 = layers.BatchNormalization()(add4)

        stacked_rnn_outputs = tf.reshape(batchnorm5, [-1, batchnorm5.shape[2]])
        stacked_outputs = keras.layers.Dense(units=1)(stacked_rnn_outputs)
        logits = tf.reshape(stacked_outputs, [-1, n_timestep, 1])

        return Model(inputs=inputs, outputs=logits)


def CALNN(time_step, n_input, lstm_uits):
    with tf.device('/gpu:0'):
        inputs = keras.Input(shape=(time_step, n_input, 1))
        conv1 = ConvLayer(inputs)

        # conv1 = ConvLayer()(inputs)
    with tf.device('/gpu:1'):
        conv2 = ConvLayer(conv1)
        CNN_output = tf.reshape(conv2, (-1, conv2.shape[1], conv2.shape[2] * conv2.shape[3]))

    with tf.device('/gpu:2'):
        attn = AttentionLayer(CNN_output.shape[2])([CNN_output, CNN_output, CNN_output])

    with tf.device('/gpu:3'):
        lstm1 = LSTMLayer(lstm_uits)(attn)
        lstm2 = LSTMLayer(lstm_uits)(lstm1)
        lstm3 = LSTMLayer(lstm_uits)(lstm2)

        stacked_rnn_outputs = tf.reshape(lstm3, [-1, lstm_uits])
        stacked_outputs = keras.layers.Dense(units=1)(stacked_rnn_outputs)
        logits = tf.reshape(stacked_outputs, [-1, time_step, 1])

    return Model(inputs=inputs, outputs=logits)


def multi_input_CLNN(time_step, n_input1, n_input2, n_input3, lstm_uits):
    with tf.device('/gpu:0'):
        inputs1 = keras.Input(shape=(time_step, n_input1, 1))
        conv1_1 = ConvLayer()(inputs1)
        conv1_2 = ConvLayer()(conv1_1)
        conv1_output = tf.reshape(conv1_2, (-1, conv1_2.shape[1], conv1_2.shape[2] * conv1_2.shape[3]))

    with tf.device('/gpu:1'):
        inputs2 = keras.Input(shape=(time_step, n_input2, 1))
        conv2_1 = ConvLayer()(inputs2)
        conv2_2 = ConvLayer()(conv2_1)
        conv2_output = tf.reshape(conv2_2, (-1, conv2_2.shape[1], conv2_2.shape[2] * conv2_2.shape[3]))

    with tf.device('/gpu:2'):
        inputs3 = keras.Input(shape=(time_step, n_input3, 1))
        conv3_1 = ConvLayer()(inputs3)
        conv3_2 = ConvLayer()(conv3_1)
        conv3_output = tf.reshape(conv3_2, (-1, conv3_2.shape[1], conv3_2.shape[2] * conv3_2.shape[3]))

    with tf.device('/gpu:3'):
        cnn_output = tf.keras.layers.concatenate(inputs=[conv1_output, conv2_output, conv3_output], axis=-1)

        lstm1 = LSTMLayer(lstm_uits)(cnn_output)
        lstm2 = LSTMLayer(lstm_uits)(lstm1)
        lstm3 = LSTMLayer(lstm_uits)(lstm2)

        stacked_rnn_outputs = tf.reshape(lstm3, [-1, lstm_uits])
        stacked_outputs = keras.layers.Dense(units=1)(stacked_rnn_outputs)
        logits = tf.reshape(stacked_outputs, [-1, time_step, 1])
    return Model(inputs=[inputs1, inputs2, inputs3], outputs=logits)


class PoolingAttentionLayer(Model):
    def __init__(self, size_group):
        super(PoolingAttentionLayer, self).__init__()
        self.size_group = size_group

    def call(self, inputs, training=False):
        self.groups = list()
        self.attn_li = list()
        value = inputs[0]  # conv_layer_output
        target = inputs[1]  # target_label
        target = tf.transpose(target, [0, 3, 1, 2])  # (batch, filter, time, feature =1)
        n_group = int(int(value.shape[2]) / self.size_group)  # number of group

        remainder = int(value.shape[2]) % self.size_group
        for i in range(0, n_group):  # split group
            self.groups.append(tf.slice(value, [0, 0, i * self.size_group, 0], [-1, -1, self.size_group, -1]))

        if remainder != 0:
            self.groups.append(tf.slice(value, [0, 0, n_group + 1 * self.size_group, 0], [-1, -1, remainder, -1]))

        for group in self.groups:  # compute attention by group
            group_trans = tf.transpose(group, [0, 3, 2, 1])
            score = tf.matmul(group_trans, target)
            attn = tf.nn.softmax(score)
            self.attn_li.append(tf.matmul(tf.transpose(group, [0, 3, 1, 2]), attn))

        result = tf.concat(self.attn_li, axis=3)  # concat group

        result = tf.transpose(result, [0, 2, 3, 1])  # (batch, time,fature, filter)
        return result


def PoolingAttention(time_step, n_input, lstm_uits, gpu):
    with tf.device('/gpu:' + str(gpu)):
        inputs = keras.Input(shape=(time_step, n_input, 1))
        target = keras.Input(shape=(time_step, 1, 1))

        # target = np.expand_dims(target, axis = -1)
        conv1 = ConvLayer(inputs, drop_rate=0.8, filters=8, kernel_size=(1, 100), strides=(1, 1), padding='SAME')

        attn1 = PoolingAttentionLayer(10)([conv1, target])
        print('conv1 shape :', conv1.shape, '  target shape : ', target.shape)
        print('attn1 shape : ', attn1.shape)
        conv2 = ConvLayer(attn1, drop_rate=0.8, filters=8, kernel_size=(1, 100), strides=(1, 1), padding='SAME')
        print('conv2 shape :', conv2.shape, '  target shape :', target.shape)
        attn2 = PoolingAttentionLayer(10)([conv2, target])
        CNN_output = tf.reshape(attn2, (-1, attn2.shape[1], attn2.shape[2] * attn2.shape[3]))

        lstm1 = layers.LSTM(lstm_uits,
                            return_sequences=True,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(0.01))(CNN_output)
        lstm2 = layers.LSTM(lstm_uits,
                            return_sequences=True,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(0.01))(lstm1)
        lstm3 = layers.LSTM(lstm_uits,
                            return_sequences=True,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(0.01))(lstm2)

        stacked_rnn_outputs = tf.reshape(lstm3, [-1, lstm_uits])
        stacked_outputs = keras.layers.Dense(units=1)(stacked_rnn_outputs)
        logits = tf.reshape(stacked_outputs, [-1, time_step, 1])

    return Model(inputs=[inputs, target], outputs=logits)


def AutoEncoderLSTM(n_timestep, n_inputs, coding_size, regularizers_alpha=0.01, drop_rate=0.5, gpu=0):
    with tf.device('/gpu:' + str(gpu)):
        inputs = keras.Input(shape=(n_timestep, n_inputs))
        dropout = layers.Dropout(rate=drop_rate)(inputs)
        lstm1 = layers.LSTM(1024,
                            return_sequences=True,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizers_alpha), name='HiddenLayer1')(dropout)

        lstm2 = layers.LSTM(coding_size,
                            return_sequences=True,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizers_alpha), name='HiddenLayer2')(lstm1)

        lstm3 = layers.LSTM(1024,
                            return_sequences=True,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizers_alpha), name='HiddenLayer3')(lstm2)

        lstm4 = layers.LSTM(n_inputs,
                            return_sequences=True,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(lstm3)

        return Model(inputs=inputs, outputs=lstm4)


def AutoEncoderDNN(n_item, n_inputs, n_units, regularizers_alpha=0.01, drop_rate=0.5, gpu=0):
    with tf.device('/gpu:' + str(gpu)):
        inputs = keras.Input(shape=(n_item, n_inputs))
        dropout = layers.Dropout(rate=drop_rate)(inputs)
        dnn1 = layers.Dense(n_units[0],
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizers_alpha), name='HiddenLayer1')(dropout)

        dnn2 = layers.Dense(n_units[1],
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizers_alpha), name='HiddenLayer2')(dnn1)

        dnn3 = layers.Dense(n_units[0],
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizers_alpha), name='HiddenLayer3')(dnn2)

        dnn4 = layers.Dense(n_inputs,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(dnn3)

        return Model(inputs=inputs, outputs=dnn4)


def next_batch(X_data, y_data, batch_size):
    idx = np.random.choice(len(y_data), batch_size, replace=False)
    return X_data[idx], y_data[idx]


def next_batch2(X_data, y_data, target, batch_size):
    idx = np.random.choice(len(y_data), batch_size, replace=False)
    return X_data[idx], y_data[idx], target[idx]


def next_batch_multi_input(input1, input2, input3, y_data, batch_size):
    idx = np.random.choice(len(y_data), batch_size, replace=False)
    return input1[idx], input2[idx], input3[idx], y_data[idx]


def loss_function(model, input_data, output_data):
    logits = model(input_data, training=True)
    loss = tf.reduce_mean(tf.square(logits - output_data))
    return loss


def loss_fn(model, images, labels):
    logits = model(images, training=True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=labels))
    return loss


def gradient(model, input_data, output_data):
    with tf.GradientTape() as tape:
        loss = loss_function(model, input_data, output_data)

    return tape.gradient(loss, model.trainable_variables), loss


def evaluate(model, input_data, output_data, dropout=False):
    logits = model(input_data, training=dropout).numpy()
    #MSE = tf.reduce_mean(tf.square(logits - output_data))
    MSE = np.around(np.mean(np.square(logits - output_data)),decimals=4)
    return MSE

"""
def training_backup(model, train_input, train_output, test_input, test_output, n_iteration, batch_size, learning_rate,
                    encoding=None):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for iteration in range(n_iteration):
        batch_train_input, batch_train_output = models.next_batch(train_input, train_output, batch_size)
        batch_test_input, batch_test_output = models.next_batch(test_input, test_output, batch_size)

        if encoding != None:
            batch_train_input = encoding(batch_train_input, training=False)
            batch_test_input = encoding(batch_test_input, training=False)

        gradients = models.gradient(model, batch_train_input, batch_train_output)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if iteration % 100 == 0:
            loss = models.loss_function(model, batch_train_input, batch_train_output)
            train_MSE = models.evaluate(model, batch_train_input, batch_train_output)
            test_MSE = models.evaluate(model, batch_test_input, batch_test_output)

            print('iteration :', iteration, ' loss =', loss.numpy(), ' train MSE =', train_MSE.numpy(), ' test MSE =',
                  test_MSE.numpy())
    return model
"""

def training(model, train_input, train_output, test_input, test_output, n_iteration, train_batch_size, test_batch_size,
             learning_rate, encoding=None):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for iteration in range(n_iteration):
        batch_train_input, batch_train_output = next_batch(train_input, train_output, train_batch_size)
        # batch_test_input, batch_test_output = .next_batch(test_input, test_output, test_batch_size)

        if encoding != None:
            batch_train_input = encoding(batch_train_input, training=False)
            test_input = encoding(test_input, training=False)

        gradients, loss = gradient(model, batch_train_input, batch_train_output)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if iteration % 100 == 0:
            # loss = loss_function(model, batch_train_input, batch_train_output)
            train_MSE = evaluate(model, batch_train_input, batch_train_output)
            test_MSE = evaluate(model, test_input, test_output)

            print('iteration :', iteration, ' loss =', loss.numpy(), ' train MSE =', train_MSE, ' test MSE =',
                  test_MSE)
    return model

# ==================== Transformer Components ====================

class PositionalEncoding(layers.Layer):
    """위치 인코딩 레이어 - 시계열 데이터의 순서 정보를 추가"""
    def __init__(self, max_sequence_length=300, d_model=256):
        super(PositionalEncoding, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        
    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles
    
    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        
        # 위치 인덱스 생성
        position = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(self.d_model, dtype=tf.float32)[tf.newaxis, :]
        
        # 각도 계산
        angle_rads = self.get_angles(position, i, self.d_model)
        
        # sin을 짝수 인덱스에, cos를 홀수 인덱스에 적용
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        
        # 교차로 합치기
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return inputs + pos_encoding


class MultiHeadAttention(layers.Layer):
    """멀티 헤드 어텐션 레이어"""
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.depth = d_model // num_heads
        
        # Query, Key, Value를 위한 Dense 레이어
        self.wq = layers.Dense(d_model, kernel_initializer='glorot_uniform')
        self.wk = layers.Dense(d_model, kernel_initializer='glorot_uniform')
        self.wv = layers.Dense(d_model, kernel_initializer='glorot_uniform')
        
        self.dense = layers.Dense(d_model, kernel_initializer='glorot_uniform')
        self.dropout = layers.Dropout(dropout_rate)
        
    def split_heads(self, x, batch_size):
        """마지막 차원을 (num_heads, depth)로 분할"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, query, key, value, mask=None, training=False):
        batch_size = tf.shape(query)[0]
        
        # Query, Key, Value 생성
        q = self.wq(query)  # (batch_size, seq_len, d_model)
        k = self.wk(key)
        v = self.wv(value)
        
        # 헤드 분할
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled Dot-Product Attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # 스케일링
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # 마스크 적용 (선택적)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Softmax로 어텐션 가중치 계산
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)
        
        # Value에 가중치 적용
        output = tf.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q, depth)
        
        # 헤드 병합
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        
        # 최종 선형 변환
        output = self.dense(concat_attention)
        
        return output, attention_weights


class FeedForwardNetwork(layers.Layer):
    """피드포워드 네트워크"""
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.dense1 = layers.Dense(dff, activation='relu', kernel_initializer='he_normal')
        self.dense2 = layers.Dense(d_model, kernel_initializer='he_normal')
        self.dropout = layers.Dropout(dropout_rate)
        
    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x


class TransformerEncoderLayer(layers.Layer):
    """Transformer 인코더 레이어"""
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.ffn = FeedForwardNetwork(d_model, dff, dropout_rate)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, x, mask=None, training=False):
        # 멀티 헤드 어텐션
        attn_output, _ = self.mha(x, x, x, mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # Residual connection
        
        # 피드포워드 네트워크
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # Residual connection
        
        return out2


# ==================== Transformer Models ====================

def VanillaTransformer(n_timestep, n_inputs, d_model=256, num_heads=8, num_layers=4, 
                       dff=1024, dropout_rate=0.1, gpu=0):
    """
    기본 Transformer 모델
    
    Args:
        n_timestep: 시계열 길이 (예: 60일)
        n_inputs: 입력 특징 개수
        d_model: 모델 차원 (256)
        num_heads: 어텐션 헤드 개수 (8)
        num_layers: 인코더 레이어 개수 (4)
        dff: 피드포워드 네트워크 차원 (1024)
        dropout_rate: 드롭아웃 비율 (0.1)
        gpu: 사용할 GPU 번호
    
    Returns:
        Keras Model
    """
    with tf.device('/gpu:' + str(gpu)):
        # 입력
        inputs = keras.Input(shape=(n_timestep, n_inputs))
        
        # 입력 차원을 d_model로 변환
        x = layers.Dense(d_model, kernel_initializer='he_normal')(inputs)
        
        # 위치 인코딩 추가
        pos_encoding = PositionalEncoding(max_sequence_length=n_timestep, d_model=d_model)
        x = pos_encoding(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Transformer 인코더 레이어들
        for _ in range(num_layers):
            x = TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)(x)
        
        # 출력 레이어 - 각 시간스텝마다 예측값 생성
        outputs = layers.Dense(1, kernel_initializer='he_normal')(x)
        
        return Model(inputs=inputs, outputs=outputs, name='VanillaTransformer')


def TransformerRegressor(n_timestep, n_inputs, d_model=256, num_heads=8, num_layers=4,
                         dff=1024, dropout_rate=0.1, gpu=0):
    """
    시계열 회귀를 위한 Transformer 모델 (GlobalAveragePooling 사용)
    마지막 출력만 사용하는 경우에 적합
    
    Args:
        n_timestep: 시계열 길이
        n_inputs: 입력 특징 개수
        d_model: 모델 차원
        num_heads: 어텐션 헤드 개수
        num_layers: 인코더 레이어 개수
        dff: 피드포워드 네트워크 차원
        dropout_rate: 드롭아웃 비율
        gpu: 사용할 GPU 번호
    
    Returns:
        Keras Model
    """
    with tf.device('/gpu:' + str(gpu)):
        inputs = keras.Input(shape=(n_timestep, n_inputs))
        
        # 입력 임베딩
        x = layers.Dense(d_model, kernel_initializer='he_normal')(inputs)
        
        # 위치 인코딩
        pos_encoding = PositionalEncoding(max_sequence_length=n_timestep, d_model=d_model)
        x = pos_encoding(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Transformer 인코더
        for _ in range(num_layers):
            x = TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)(x)
        
        # Global Average Pooling - 모든 시간스텝의 평균
        x = layers.GlobalAveragePooling1D()(x)
        
        # 출력 레이어
        x = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1, kernel_initializer='he_normal')(x)
        
        return Model(inputs=inputs, outputs=outputs, name='TransformerRegressor')


def AutoEncoderTransformer(n_timestep, n_inputs, coding_size=256, num_heads=8, 
                           num_encoder_layers=3, num_decoder_layers=3,
                           dff=1024, dropout_rate=0.1, gpu=0):
    """
    AutoEncoder 구조의 Transformer
    입력 데이터를 압축 후 복원
    
    Args:
        n_timestep: 시계열 길이
        n_inputs: 입력 특징 개수
        coding_size: 압축 차원
        num_heads: 어텐션 헤드 개수
        num_encoder_layers: 인코더 레이어 개수
        num_decoder_layers: 디코더 레이어 개수
        dff: 피드포워드 네트워크 차원
        dropout_rate: 드롭아웃 비율
        gpu: 사용할 GPU 번호
    
    Returns:
        Keras Model
    """
    with tf.device('/gpu:' + str(gpu)):
        inputs = keras.Input(shape=(n_timestep, n_inputs))
        
        # === Encoder ===
        x = layers.Dense(512, kernel_initializer='he_normal')(inputs)
        pos_encoding = PositionalEncoding(max_sequence_length=n_timestep, d_model=512)
        x = pos_encoding(x)
        x = layers.Dropout(dropout_rate)(x)
        
        for _ in range(num_encoder_layers):
            x = TransformerEncoderLayer(512, num_heads, dff, dropout_rate)(x)
        
        # 압축 레이어 (Bottleneck)
        encoded = layers.Dense(coding_size, kernel_initializer='he_normal', name='HiddenLayer')(x)
        
        # === Decoder ===
        x = layers.Dense(512, kernel_initializer='he_normal')(encoded)
        
        for _ in range(num_decoder_layers):
            x = TransformerEncoderLayer(512, num_heads, dff, dropout_rate)(x)
        
        # 복원 레이어
        outputs = layers.Dense(n_inputs, kernel_initializer='he_normal')(x)
        
        return Model(inputs=inputs, outputs=outputs, name='AutoEncoderTransformer')


def HybridTransformerLSTM(n_timestep, n_inputs, d_model=256, num_heads=8, 
                          num_transformer_layers=2, lstm_units=256,
                          dropout_rate=0.1, gpu=0):
    """
    Transformer + LSTM 하이브리드 모델
    Transformer로 장기 의존성을 포착하고 LSTM으로 시계열 패턴 학습
    
    Args:
        n_timestep: 시계열 길이
        n_inputs: 입력 특징 개수
        d_model: Transformer 모델 차원
        num_heads: 어텐션 헤드 개수
        num_transformer_layers: Transformer 레이어 개수
        lstm_units: LSTM 유닛 개수
        dropout_rate: 드롭아웃 비율
        gpu: 사용할 GPU 번호
    
    Returns:
        Keras Model
    """
    with tf.device('/gpu:' + str(gpu)):
        inputs = keras.Input(shape=(n_timestep, n_inputs))
        
        # Transformer 레이어
        x = layers.Dense(d_model, kernel_initializer='he_normal')(inputs)
        pos_encoding = PositionalEncoding(max_sequence_length=n_timestep, d_model=d_model)
        x = pos_encoding(x)
        x = layers.Dropout(dropout_rate)(x)
        
        for _ in range(num_transformer_layers):
            x = TransformerEncoderLayer(d_model, num_heads, d_model * 4, dropout_rate)(x)
        
        # LSTM 레이어
        x = layers.LSTM(lstm_units, return_sequences=True, 
                       kernel_initializer='he_normal',
                       kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.LSTM(lstm_units, return_sequences=True,
                       kernel_initializer='he_normal',
                       kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # 출력 레이어
        outputs = layers.Dense(1, kernel_initializer='he_normal')(x)
        
        return Model(inputs=inputs, outputs=outputs, name='HybridTransformerLSTM')


def ResTransformer(n_timestep, n_inputs, d_model=256, num_heads=8, num_layers=6,
                   dff=1024, dropout_rate=0.1, gpu=0):
    """
    깊은 Residual Connection을 가진 Transformer
    더 깊은 네트워크 학습 가능
    
    Args:
        n_timestep: 시계열 길이
        n_inputs: 입력 특징 개수
        d_model: 모델 차원
        num_heads: 어텐션 헤드 개수
        num_layers: 인코더 레이어 개수
        dff: 피드포워드 네트워크 차원
        dropout_rate: 드롭아웃 비율
        gpu: 사용할 GPU 번호
    
    Returns:
        Keras Model
    """
    with tf.device('/gpu:' + str(gpu)):
        inputs = keras.Input(shape=(n_timestep, n_inputs))
        
        # 입력 임베딩
        x = layers.Dense(d_model, kernel_initializer='he_normal')(inputs)
        initial = x  # Residual connection을 위해 저장
        
        # 위치 인코딩
        pos_encoding = PositionalEncoding(max_sequence_length=n_timestep, d_model=d_model)
        x = pos_encoding(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Transformer 레이어들 with skip connections
        for i in range(num_layers):
            x_prev = x
            x = TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)(x)
            
            # 매 2개 레이어마다 skip connection 추가
            if i % 2 == 1:
                x = layers.Add()([x, x_prev])
        
        # 초기 입력과의 residual connection
        x = layers.Add()([x, initial])
        
        # 출력 레이어
        outputs = layers.Dense(1, kernel_initializer='he_normal', name='logits')(x)
        
        return Model(inputs=inputs, outputs=outputs, name='ResTransformer')

