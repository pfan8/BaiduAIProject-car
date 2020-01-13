import sys
sys.path.insert(1, '.')
from utils.data_loader import preprocess_sentence
from utils.wv_loader import load_vocab,load_embedding_matrix
import utils.config as config
from model_tf.layers import Encoder, Decoder
import matplotlib
from matplotlib import font_manager
import matplotlib.ticker as ticker
# 解决中文乱码
font=font_manager.FontProperties(fname="data/TrueType/simhei.ttf")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import tensorflow as tf
import os

input_sequence_len = 300
EPOCHS = 10
BATCH_SIZE = 64
embedding_dim = 300
units = 256

# 输入的长度
max_length_inp = 300
# 输出的长度
max_length_targ = 55

word2id, id2word = load_vocab()
vocab_size = len(word2id)
# 使用GenSim训练好的embedding matrix
embedding_matrix = load_embedding_matrix()
test_X = np.loadtxt(config.test_x_path)

def evaluate(sentence, encoder, decoder):
    attention_plot = np.zeros((max_length_targ, max_length_inp + 2))

    # inputs = preprocess_sentence(sentence, max_length_inp, word2id)
    inputs = np.expand_dims(test_X[0], axis=0)

    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([word2id['<START>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1,))

        attention_plot[t] = attention_weights.numpy()
        predicted_id = tf.argmax(predictions[0]).numpy()

        result += id2word[predicted_id] + ' '
        if id2word[predicted_id] == '<STOP>':
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot

# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14,'fontproperties':font}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def translate(sentence, encoder, decoder):
    result, sentence, attention_plot = evaluate(sentence, encoder, decoder)

    sentence = sentence.replace(' <PAD>', '')
    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))

if __name__ == '__main__':
    encoder = Encoder(vocab_size, embedding_dim, embedding_matrix, units, BATCH_SIZE)
    decoder = Decoder(vocab_size, embedding_dim, embedding_matrix, units, BATCH_SIZE)
    optimizer = tf.keras.optimizers.Adam()
    checkpoint_dir = 'data/checkpoints/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    sentence = ' '.join([id2word[w] for w in test_X[0]])
    print(sentence)
    translate(sentence, encoder, decoder)

