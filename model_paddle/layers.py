import numpy as np
import pandas as pd
import re
import jieba
from multiprocessing import cpu_count, Pool
from utils.utils import *
import paddle.fluid as fluid

class Encoder(fluid.dygraph.Layer):
    def __init__(self, 
                 name_scope, 
                 enc_units, 
                 batch_size, 
                 vocab_size=1e5, 
                 word_vector=None,
                 param_attr=None,
                 bias_attr=None,
                 is_reverse=False,
                 gate_activation='sigmoid',
                 candidate_activation='tanh',
                 h_0=None,
                 origin_mode=False):
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
        if self.word_vector is not None:
            self.vocab_size = int(self.word_vector.shape[0])
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
        
        # GRU
        self._gru = fluid.dygraph.GRUUnit(
            self.full_name(),
            size=self.enc_units * 3,
            param_attr=param_attr,
            bias_attr=bias_attr,
            activation=candidate_activation,
            gate_activation=gate_activation,
            origin_mode=origin_mode)
        self.h_0 = h_0 if h_0 is not None else self.initialize_hidden_state()
        self.is_reverse = is_reverse
                                                
    
    def forward(self, inputs, hidden=None):
        '''
        调用Encoder时的计算
        :param inputs: variable类型的输入数据，维度（ N, T, D ）
        :param hidden: 隐藏层输入h_0
        :return output,state: output = hidden拼接向量，维度（ N, T, H ）
                              state = hidden时间维度的最后一个向量
        '''
        hidden = self.h_0 if hidden is None else hidden
        res = []
        for i in range(inputs.shape[1]):
            if self.is_reverse:
                i = inputs.shape[1] - 1 - i
            input_ = inputs[:, i:i + 1, :]
            input_ = fluid.layers.reshape(
                input_, [-1, input_.shape[2]], inplace=False)
            input_ = self._embedding(inputs)
            if hasattr(self, '_fc'):
                input_ = self._fc(input_)
            hidden, reset, gate = self._gru(input_, hidden)
            hidden_ = fluid.layers.reshape(
                hidden, [-1, 1, hidden.shape[1]], inplace=False)
            res.append(hidden_)
        if self.is_reverse:
            res = res[::-1]
        res = fluid.layers.concat(res, axis=1)
        return res, res[:,-1,:]
    
    def initialize_hidden_state(self):
        return fluid.layers.zeros((self.batch_size, self.enc_units), dtype='float32')

class BahdanauAttention(fluid.dygraph.Layer):
    def __init__(self, name_scope, units):
        super(BahdanauAttention, self).__init__(name_scope)
        self.W1 = fluid.dygraph.FC('attention_fc1', units, num_flatten_dims=2)
        self.W2 = fluid.dygraph.FC('attention_fc2', units, num_flatten_dims=2)
        self.V = fluid.dygraph.FC('attention_v', 1, num_flatten_dims=2)

    def forward(self, query, values):
        # query shape == (batch_size, 1, hidden_size)
        # values shape == (batch_size, max_length, hidden_size)
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        temp = (self.W1(values) + self.W2(query))
        score = self.V(fluid.layers.tanh(temp))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = fluid.layers.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        # paddlepaddle will not do broadcast autom atically, use elementwise_mul API
        context_vector = fluid.layers.elementwise_mul(values, attention_weights)
        context_vector = fluid.layers.reduce_sum(context_vector, dim=1)

        return context_vector, attention_weights

class Decoder(fluid.dygraph.Layer):
    def __init__(self, 
                 name_space, 
                 dec_units,
                 batch_size,
                 vocab_size=1e5, 
                 word_vector=None,
                 param_attr=None,
                 bias_attr=None,
                 is_reverse=False,
                 gate_activation='sigmoid',
                 candidate_activation='tanh',
                 h_0=None,
                 origin_mode=False):
        super(Decoder, self).__init__(name_space)
        self.batch_size = int(batch_size)
        self.dec_units = int(dec_units)
        self.vocab_size = int(vocab_size)
        self.word_vector = word_vector
        
       # Embedding
        if self.word_vector is not None:
            self.vocab_size = int(self.word_vector.shape[0])
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
        else:
            self._embedding = fluid.dygraph.Embedding(
                                name_scope='embedding',
                                size=[self.vocab_size, self.dec_units*3],
                                param_attr='emb.w',
                                is_sparse=False)
        
        # GRU
        self._gru = fluid.dygraph.GRUUnit(
            self.full_name(),
            size=self.dec_units * 3,
            param_attr=param_attr,
            bias_attr=bias_attr,
            activation=candidate_activation,
            gate_activation=gate_activation,
            origin_mode=origin_mode)
        
        # 如果维度不符合 D*3 不能传导，需要添加一层FC，保证维度是 D*3
        self._fc4gru = fluid.dygraph.FC('fc_for_gru', self.dec_units*3)
        
        # FC (N,H)==>(N,V)
        self._fc = fluid.dygraph.FC('fc', self.vocab_size)
        self.is_reverse = is_reverse
        
    def forward(self, inputs, hidden, context_vector):
        # enc_output shape == (batch_size, max_length, hidden_size)
        # initial hidden is the last step of enc_output, hidden shape == (batch_size, hidden_size)
        inputs = self._embedding(inputs)
        # after concat shape is (batch_size, context_size + embedding_size)
        inputs = fluid.layers.concat([context_vector, inputs], axis=-1)
        inputs = self._fc4gru(inputs)
        
        hidden, reset, gate = self._gru(inputs, hidden)

        output = self._fc(hidden)
        
        return output, hidden