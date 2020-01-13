from model_paddle.layers import Encoder,Decoder,BahdanauAttention
import utils.wv_loader as wv_loader
import paddle.fluid as fluid

class Seq2Seq(fluid.dygraph.Layer):
    def __init__(self, name_scope, params):
        super(Seq2Seq, self).__init__(name_scope)
        self.embedding_matrix = wv_loader.load_embedding_matrix()
        self.params = params
        self.encoder = Encoder('encoder',
                               enc_units   = params["enc_units"],
                               batch_size  = params["batch_size"],
                               word_vector = self.embedding_matrix)

        self.attention = BahdanauAttention('attention', params["attn_units"])

        self.decoder = Decoder('decoder',
                               dec_units   = params["dec_units"],
                               batch_size  = params["batch_size"],
                               word_vector = self.embedding_matrix)

    def call_encoder(self, enc_inp):
        enc_hidden = self.encoder.initialize_hidden_state()
        enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)
        return enc_output, enc_hidden

    def call_decoder_onestep(self, dec_input, dec_hidden, enc_output):
        context_vector, attention_weights = self.attention(dec_hidden, enc_output)

        pred, dec_hidden = self.decoder(dec_input,
                                        None,
                                        context_vector)
        return pred, dec_hidden, context_vector, attention_weights

    def forward(self, dec_hidden, enc_output, dec_target):
        predictions = []
        attentions = []
        
        # print('enc_output shape:{}'.format(enc_output.shape))
        context_vector, attn = self.attention(fluid.layers.reshape(dec_hidden,[dec_hidden.shape[0],1,-1])
                                                    , enc_output)
        dec_input = fluid.layers.reshape(dec_target[:, 0], [dec_target.shape[0],1,-1])

        for t in range(1, dec_target.shape[1]):
            pred, dec_hidden = self.decoder(dec_input,
                                            dec_hidden,
                                            context_vector)

            context_vector, attn = self.attention(fluid.layers.reshape(dec_hidden,[dec_hidden.shape[0],1,-1])
                                                    , enc_output)
            # using teacher forcing
            dec_input = fluid.layers.reshape(dec_target[:, t], [dec_target.shape[0],1,-1])

            predictions.append(fluid.layers.reshape(pred, [pred.shape[0],1,-1]))
            attentions.append(fluid.layers.reshape(attn, [attn.shape[0],1,-1]))

        predictions = fluid.layers.concat(predictions, axis=1)
        attentions = fluid.layers.concat(attentions, axis=1)

        return predictions, attentions 