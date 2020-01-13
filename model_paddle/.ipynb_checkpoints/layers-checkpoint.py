import numpy as np
import pandas as pd
import re
import jieba
from multiprocessing import cpu_count, Pool
from utils.utils import *
import paddle.fluid as fluid

class Encoder(fluid.dygraph.Layer):
    def __init__(self, name_scope, enc_units, batch_size, vocab_size=1e5, word_vector=None):
        '''
            Encoder初始化
            :param name_scope: 所在命名空间
            :param enc_units:GRU单元维度
            :param batch_size: Batch数量
            :param wordvector: 自定义词向量
            
        '''
        super(Encoder, self).__init__(name_scope)
        self.enc_units = int(enc_units)
        self.batch_size = int(batch_size)
        self.vocab_size = int(vocab_size)
        self.word_vector = word_vector
        
        # Embedding
        if word_vector is not None:
            w_param_attrs = fluid.ParamAttr(
                                name="emb_weight",
                                learning_rate=0.5,
                                initializer=fluid.initializer.NumpyArrayInitializer(self.word_vector),
                                trainable=True)
            self._embedding = fluid.dygraph.Embedding(
                                name_scope='embedding',
                                size=list(self.word_vector.shape),
                                param_attr= w_param_attrs,
                                is_sparse=False)
            # 如果有自定义词向量维度不符合 D*3，需要添加一层FC
            if self.word_vector.shape[1] != self.enc_units*3:
                self._fc = fluid.dygraph.FC('fc_for_gru', self.enc_units*3)
        else:
            self._embedding = fluid.dygraph.Embedding(
                                name_scope='embedding',
                                size=[self.vocab_size, self.enc_units*3],
                                param_attr='emb.w',
                                is_sparse=False)
        
        self._gru = fluid.dygraph.GRUUnit(name_scope='gru', size=self.enc_units*3)
                                                
    
    def forward(self, inputs, hidden):
        '''
        调用Encoder时的计算
        :param inputs: variable类型的输入数据
        :param hidden: 隐藏层输入h_0
        '''
        x = self._embedding(inputs)
        if hasattr(self, '_fc'):
            x = self._fc(x)
        x = self._gru(x, hidden)
        return x
    
    def initialize_hidden_state(self):
        return fluid.layers.zeros((self.batch_size, self.enc_units), dtype='float32')

class Decoder(fluid.dygraph.Layer):
    def __init__(self, name_space, dec_units, batch_size, vocab_size=1e5, word_vector=None):
        super(Decoder, self).__init__(name_space)
        self.batch_size = int(batch_size)
        self.dec_units = int(dec_units)
        self.vocab_size = int(vocab_size)
        self.word_vector = word_vector
        
       # Embedding
        if word_vector is not None:
            w_param_attrs = fluid.ParamAttr(
                                name="emb_weight",
                                learning_rate=0.5,
                                initializer=fluid.initializer.NumpyArrayInitializer(self.word_vector),
                                trainable=True)
            self._embedding = fluid.dygraph.Embedding(
                                name_scope='embedding',
                                size=list(self.word_vector.shape),
                                param_attr= w_param_attrs,
                                is_sparse=False)
            # 如果有自定义词向量维度不符合 D*3，需要添加一层FC
            if self.word_vector.shape[1] != self.dec_units*3:
                self._fc4gru = fluid.dygraph.FC('fc_for_gru', self.dec_units*3)
        else:
            self._embedding = fluid.dygraph.Embedding(
                                name_scope='embedding',
                                size=[self.vocab_size, self.dec_units*3],
                                param_attr='emb.w',
                                is_sparse=False)
        
        self._gru = fluid.dygraph.GRUUnit(name_scope='gru', size=self.dec_units*3)
        # FC (N,H)==>(N,V)
        self._fc = fluid.dygraph.FC('fc', self.vocab_size)

        # Attention还未添加，先留着
        # used for attention
#         self.attention = BahdanauAttention(self.dec_units)

    def forward(self, enc_output, hidden):
        # enc_output shape == (batch_size, max_length, hidden_size)
#         context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self._embedding(enc_output)
        if hasattr(self, '_fc4gru'):
            x = self._fc4gru(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
#         x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, rhp, state = self._gru(x, hidden)

        # output shape == (batch_size, vocab)
        x = self._fc(output)

#         return x, state, attention_weights
        return x, state

# 测试Stab
if __name__ == '__main__':
    emb_matrix = load_file('output/fasttext_emb.pkl')
    use_emb = True
    with fluid.dygraph.guard():
        # Encoder
        enc_units = 64
        batch_size = 100
        if use_emb:
            encoder = Encoder('encoder', enc_units=enc_units, batch_size=batch_size, word_vector=emb_matrix)
        else:
            encoder = Encoder('encoder', enc_units=enc_units, batch_size=batch_size)
        # 生成一个时间步的随机输入，维度为(B,1)
        X = fluid.dygraph.base.to_variable(np.random.randint(emb_matrix.shape[0], size=(batch_size, 1))) # 假设在单个时间点t上
        hidden = encoder.initialize_hidden_state()
        result_enc = encoder(X, hidden)
        for var in result_enc:
            print("result_enc shape is {}".format(var.numpy().shape))
        # Decoder
        dec_units = 64
        batch_size = 100
        if use_emb:
            decoder = Decoder('decoder', dec_units=dec_units, batch_size=batch_size, word_vector=emb_matrix)
        else:
            decoder = Decoder('decoder', dec_units=dec_units, batch_size=batch_size)
        X = fluid.dygraph.base.to_variable(np.random.randint(emb_matrix.shape[0], size=(batch_size, 1))) # 假设在单个时间点t上
        hidden = result_enc[0]
        result_dec, state = decoder(X, hidden)
        # 词典大小假设是10万
        print("result_dec shape is {}".format(result_dec.numpy().shape)) # (N, V)