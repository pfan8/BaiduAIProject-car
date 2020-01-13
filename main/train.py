# -*- coding:utf-8 -*-
# Created by LuoJie at 11/29/19
import tensorflow as tf
import os
import warnings
import sys
sys.path.insert(1, '.')

warnings.filterwarnings("ignore")
from model_tf.layers import Encoder, Decoder
from utils.data_loader import load_dataset
from utils.wv_loader import load_embedding_matrix, get_vocab
import time


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def get_checkpoint():
    checkpoint_dir = 'data/checkpoints/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    return checkpoint, checkpoint_prefix


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([vocab['<START>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # decoder(x, hidden, enc_output)
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))

        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss


def train_model(train_X, train_Y, epoch_num, BATCH_SIZE):
    EPOCHS = int(epoch_num)
    BUFFER_SIZE = len(train_X)
    checkpoint, checkpoint_prefix = get_checkpoint()
    steps_per_epoch = len(train_X) // BATCH_SIZE
    dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 5 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


# 读取vocab训练
vocab, reverse_vocab = get_vocab()

if __name__ == '__main__':
    # 计算vocab size
    vocab_size = len(vocab)
    # 加载数据集
    train_X, train_Y, test_X = load_dataset()
    # 使用GenSim训练好的embedding matrix
    embedding_matrix = load_embedding_matrix()

    input_sequence_len = 300
    EPOCHS = 10
    BATCH_SIZE = 64
    embedding_dim = 300
    units = 256
    encoder = Encoder(vocab_size, embedding_dim, embedding_matrix, units, BATCH_SIZE)
    decoder = Decoder(vocab_size, embedding_dim, embedding_matrix, units, BATCH_SIZE)

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    train_model(train_X, train_Y, EPOCHS, BATCH_SIZE)
