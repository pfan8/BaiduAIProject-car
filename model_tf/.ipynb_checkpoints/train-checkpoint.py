import sys
sys.path.insert(1, '.')
import paddle
import paddle.fluid as fluid
from utils.data_loader import load_dataset
from utils.wv_loader import load_vocab

X,Y,_ = load_dataset()
x_time_steps = X.shape[1]
y_time_steps = Y.shape[1]
word2id, id2word = load_vocab()

def train_gen():
    for row in Dataset:
        yield row

# 定义loss函数
def loss_function(real, pred):
    # 判断logit为1和0的数量
    real = fluid.layers.cast(real, dtype=np.int64)
    pad_array = fluid.layers.ones_like(real) * word2id['<PAD>']
    mask = fluid.layers.logical_not(fluid.layers.equal(real, pad_array))
    # 计算decoder的长度
    dec_lens = fluid.layers.reduce_sum(fluid.layers.cast(mask, dtype=np.float32), dim=-1)
    # 计算loss值
    loss_ = fluid.layers.cross_entropy(input=pred, label=real)
    # 转换mask的格式
    mask = fluid.layers.cast(mask, dtype=loss_.dtype)
    # 调整loss
    loss_ *= mask
    # 确认下是否有空的摘要别加入计算
    loss_ = fluid.layers.reduce_sum(loss_, dim=-1) / real.shape[0]
    return fluid.layers.reduce_mean(loss_)

if __name__ == "__main__":
    with fluid.dygraph.guard():
        epoch_num = 5
        BATCH_SIZE = 64

        # Encoder
        enc_units = 64
        encoder = Encoder('encoder'
                            , enc_units=enc_units
                            , batch_size=BATCH_SIZE
                            , word_vector=emb_matrix
                            , vocab_size=len(word2id))
        # Decoder
        dec_units = 64
        decoder = Decoder('decoder'
                            , dec_units=dec_units
                            , batch_size=BATCH_SIZE
                            , word_vector=emb_matrix
                            , attention_units=10
                            , vocab_size=len(word2id))
        # Optimizer
        adam = fluid.optimizer.Adam(learning_rate=0.001)
        train_reader = paddle.batch(
            train_gen, batch_size= BATCH_SIZE, drop_last=True)

        np.set_printoptions(precision=3, suppress=True)
        dy_param_init_value={}
        for epoch in range(epoch_num):
            print('='*25 + 'Epoch {}'.format(epoch) + '='*25)
            for batch_id, data in enumerate(train_reader()):
                if batch_id % 100 == 0:
                    print('completed batch {}'.format(batch_id))
                data = np.array(data).astype('int64').flatten().reshape(BATCH_SIZE, x_time_steps+y_time_steps, 1)
                batch_x = data[:,:x_time_steps,:]
                batch_y = data[:,x_time_steps:,:]

                input = fluid.dygraph.to_variable(batch_x)
                label = fluid.dygraph.to_variable(batch_y)

                enc_output,_ = encoder(input, encoder.initialize_hidden_state())
                # print(label.shape)
                pred,_ = decoder(label, enc_output)
                loss = loss_function(label, pred)
                avg_loss = fluid.layers.mean(loss)

                dy_out = avg_loss.numpy()

                avg_loss.backward()
                adam.minimize(avg_loss)
                if batch_id == 100:
                    fluid.dygraph.save_dygraph(encoder.state_dict(), "paddle_encoder")
                    fluid.dygraph.save_dygraph(decoder.state_dict(), "paddle_decoder")
                encoder.clear_gradients()
                decoder.clear_gradients()

                if batch_id == 100:
                    for param in encoder.parameters():
                        dy_param_init_value[param.name] = param.numpy()
                    for param in decoder.parameters():
                        dy_param_init_value[param.name] = param.numpy()
                    enc, _ = fluid.dygraph.load_dygraph("paddle_encoder")
                    dec, _ = fluid.dygraph.load_dygraph("paddle_decoder")
                    encoder.set_dict(enc)
                    decoder.set_dict(dec)
                    # break
            # if epoch == 0:
            #     break
        restore = encoder.parameters() + decoder.parameters()
        # check save and load

        success = True
        for value in restore:
            if (not np.array_equal(value.numpy(), dy_param_init_value[value.name])) or (not np.isfinite(value.numpy().all())) or (np.isnan(value.numpy().any())):
                success = False
        print("model_paddle save and load success? {}".format(success))