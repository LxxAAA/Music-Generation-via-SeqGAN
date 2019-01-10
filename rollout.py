#这个是强化学习，得到奖励有关的代码，我觉得这里也是我的重点关注对象！
# rollout 是 G的类似版本，然后在得到 reward时，用到了D ，目前知道的就是这些。
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import numpy as np
import yaml

with open("SeqGAN.yaml") as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)


class ROLLOUT(object):
    # policy rollout object for policy gradient update 
    # it takes the generator network as object, having the same structure as the generator 与G又相同的架构
    # during adversarial training, it produces rewards by get_reward() 通过 get_reward 来获取reward，给 G
    # it updates its parameters by update_params() 通过update_params来更新参数
    def __init__(self, lstm, update_rate): ###################################### 传进来的是G， 难道G就是LSTM？？ 应该是，我看了G半天还没看懂，玛德
        
        # define the network & update rate 定义网络和学习率
        self.lstm = lstm 
        self.update_rate = update_rate
        ####################################################################################################
        ###############################################             定义 lstm网络的超参数和一些元素
        # define hyperparams of the lstm network 
        self.num_emb = self.lstm.num_emb
        self.batch_size = self.lstm.batch_size
        self.emb_dim = self.lstm.emb_dim
        self.hidden_dim = self.lstm.hidden_dim
        self.sequence_length = self.lstm.sequence_length
        # copy the start token and learning rate of the generator
        self.start_token = tf.identity(self.lstm.start_token) ##2 y=x仅仅是tensor的一个简单赋值，不是定义的op，所以在图中不会形成一个节点，这样该管理器就失效了。tf.identity是返回一个一模一样新的tensor的op，这会增加一个新节点到gragh中
        self.learning_rate = self.lstm.learning_rate ##2为什么上面这个要用identity， 但是下面这个就直接赋值了?
        # define the generator embeddings & units
        self.g_embeddings = tf.identity(self.lstm.g_embeddings) ##2 给一个op，做了一个相同的节点
        self.g_recurrent_unit = self.create_recurrent_unit()  ##2 maps h_tm1 to h_t for generator ##2 映射 当前t-1的h 到当前t的h
        self.g_output_unit = self.create_output_unit()  # maps h_t to o_t (output token logits) ##2 映射 当前t的h 到当前t的输出output

        #####################################################################################################
        # placeholder definition for input sequence of tokens ##2 对于输入做了一个占位符
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length])
        self.given_num = tf.placeholder(tf.int32)

        # process the input x with embeddings
        # permutation is for [seq_length, batch_size, emb_dim] 交换次序，改变通道位置
        # the reference code does this within cpu (for memory efficiency?)
        # 用全心去感受，曼陀罗。
        # 执行x，把通道对调。利用CPU去进行
        with tf.device('/cpu:0'):
            self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.x), perm=[1, 0, 2])

        # unstack the processed_x to tensor array                                     unstack拆散的意思
        ta_emb_x = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)
        # same goes for the x without embedding, note the int32 instead of float32
        ta_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length)
        ta_x = ta_x.unstack(tf.transpose(self.x, perm=[1, 0]))
        #####################################################################################################

        # define zero initial state 起始状态
        self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
        # stack two of it?
        self.h0 = tf.stack([self.h0, self.h0])

        # define tensor array of fake data from generator
        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)

        # generation procedure consists of two phases: when i < given_num, and i > given_num
        # when current index i < given_num, use the provided tokens as the input at each time step
        def _g_recurrence_1(i, x_t, h_tm1, given_num, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)  # h_tm1 stands for hidden memory tuple
            x_tp1 = ta_emb_x.read(i)
            gen_x = gen_x.write(i, ta_x.read(i))
            return i+1, x_tp1, h_t, given_num, gen_x

        # when current index i >= given_num, start roll-out
        # use the output at t as the input at t+1
        def _g_recurrence_2(i, x_t, h_tm1, given_num, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            # define output logits, with size of [batch_size, vocab_size]
            o_t = self.g_output_unit(h_t)
            # calculate log probabilities
            # may expect numerical instability due to the direct usage of log, might cause NaN?
            log_prob = tf.log(tf.nn.softmax(o_t))
            # generate next token based on the log prob: reshape to 1D of batch_size, then cast to int
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
            # generate embedding from the next token, with size of [batch_size, emb_dim]
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)
            # write the next token to gen_x to the current index i
            gen_x = gen_x.write(i, next_token)
            return i+1, x_tp1, h_t, given_num, gen_x

        # generate gen_x from the defined recurrences above, using the while loop control ops
        # remember that TF uses static graph, requiring this special control flow for conditional branching
        i, x_t, h_tm1, given_num, self.gen_x = control_flow_ops.while_loop(
            # loop condition
            cond=lambda i, _1, _2, given_num, _4: i < given_num,
            # body function to loop
            body=_g_recurrence_1,
            # initial values to each variables
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                       self.h0, self.given_num, gen_x))

        # we only need gen_x from roll-out phase for further processing
        _, _, _, _, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_recurrence_2,
            loop_vars=(i, x_t, h_tm1, given_num, self.gen_x))

        # unstack the gen_x, with shape [seq_length, batch_size]
        self.gen_x = tf.transpose(self.gen_x.stack(), perm=[1, 0])

    
    def get_reward(self, sess, input_x, rollout_num, discriminator):
        """
        calculate rewards from policy rollout
        :param sess: TF session
        :param input_x: input data
        :param rollout_num: the number rollout for Monte Carlo search #用到了MC搜索 rollout_num是MC的搜索次数
        :param discriminator: discriminator object
        :return: rewards; list of reward at each step #是每一步的reward
        """
        # define empty rewards list, append for each time step 对每一步有一个连续的值是么，但是其实只需要一段时间，
        # 因为音乐这东西，不需要太远的间隔， 但是如何评价，如何给反馈？？？
        rewards = [] 
        # iterate over the defined rollout_num
        for i in range(rollout_num):   ##2 这个循环变量有什么用呢， 和 MC search有什么关系, 做很多遍，然后取一个平均吧
            # given_num for time step is explicitly from 1 to SEQ_LENGTH
            for given_num in range(1, config['SEQ_LENGTH']): #这个是一次把每个位置的reward都算出来是吗？ 应该是这个意思
                # define feed for generation
                feed = {self.x: input_x, self.given_num: given_num}
                # run the gen_x op defined from __init__ with feed
                samples = sess.run(self.gen_x, feed) ##2 上面这个是生成sample，生成样本集合 ，只算到gen——x，就是得到所有的完整样本，然后给D去评价
                # define new feed for discrimination   
                feed = {discriminator.input_x: samples, discriminator.dropout_keep_prob: 1.0}
                # run prediction by discriminator with feed
                ##2 下面这个是用生成的正负样本，去求得ypred for auc， 来得到reward信息。 这里是得到这个点，所有的后续样本。 
                ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
                ypred = np.array([item[1] for item in ypred_for_auc])
                # add rewards for each given_num
                if i == 0:  # initial rollout ##2 加入一个新的下标
                    rewards.append(ypred)
                else:  # from 2nd rollout, add to the existing value 给之前加入的下表，增加值。。。因为三次取平均嘛
                    rewards[given_num-1] += ypred

            # the last token reward 最后的token的reward，不用再生成了是吧。就是倒数第二轮剩的？
            feed = {discriminator.input_x: input_x, discriminator.dropout_keep_prob: 1.0}
            ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
            ypred = np.array([item[1] for item in ypred_for_auc])
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[19] += ypred #你这里的19让人很不爽呀亲！！！
        # average out the rewards, with shape [batch_size, seq_length]
        rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num) #默认就是反转图像的维度
        return rewards

    def create_recurrent_unit(self):
        # Weights and Bias for input and hidden tensor
        # copy-paste of the generator: the original paper assumes structure of rollout = generator
        self.Wi = tf.identity(self.lstm.Wi)
        self.Ui = tf.identity(self.lstm.Ui)
        self.bi = tf.identity(self.lstm.bi)

        self.Wf = tf.identity(self.lstm.Wf)
        self.Uf = tf.identity(self.lstm.Uf)
        self.bf = tf.identity(self.lstm.bf)

        self.Wog = tf.identity(self.lstm.Wog)
        self.Uog = tf.identity(self.lstm.Uog)
        self.bog = tf.identity(self.lstm.bog)

        self.Wc = tf.identity(self.lstm.Wc)
        self.Uc = tf.identity(self.lstm.Uc)
        self.bc = tf.identity(self.lstm.bc)

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def update_recurrent_unit(self):
        # Weights and Bias for input and hidden tensor
        self.Wi = self.update_rate * self.Wi + (1 - self.update_rate) * tf.identity(self.lstm.Wi)
        self.Ui = self.update_rate * self.Ui + (1 - self.update_rate) * tf.identity(self.lstm.Ui)
        self.bi = self.update_rate * self.bi + (1 - self.update_rate) * tf.identity(self.lstm.bi)

        self.Wf = self.update_rate * self.Wf + (1 - self.update_rate) * tf.identity(self.lstm.Wf)
        self.Uf = self.update_rate * self.Uf + (1 - self.update_rate) * tf.identity(self.lstm.Uf)
        self.bf = self.update_rate * self.bf + (1 - self.update_rate) * tf.identity(self.lstm.bf)

        self.Wog = self.update_rate * self.Wog + (1 - self.update_rate) * tf.identity(self.lstm.Wog)
        self.Uog = self.update_rate * self.Uog + (1 - self.update_rate) * tf.identity(self.lstm.Uog)
        self.bog = self.update_rate * self.bog + (1 - self.update_rate) * tf.identity(self.lstm.bog)

        self.Wc = self.update_rate * self.Wc + (1 - self.update_rate) * tf.identity(self.lstm.Wc)
        self.Uc = self.update_rate * self.Uc + (1 - self.update_rate) * tf.identity(self.lstm.Uc)
        self.bc = self.update_rate * self.bc + (1 - self.update_rate) * tf.identity(self.lstm.bc)

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self):
        self.Wo = tf.identity(self.lstm.Wo)
        self.bo = tf.identity(self.lstm.bo)

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def update_output_unit(self):
        self.Wo = self.update_rate * self.Wo + (1 - self.update_rate) * tf.identity(self.lstm.Wo)
        self.bo = self.update_rate * self.bo + (1 - self.update_rate) * tf.identity(self.lstm.bo)

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def update_params(self):
        # update the parameters of the generator rollout object
        # is this line necessary?: g_embeddings already initialized from __init__
        self.g_embeddings = tf.identity(self.lstm.g_embeddings)
        # update the recurrent unit and output unit
        self.g_recurrent_unit = self.update_recurrent_unit()
        self.g_output_unit = self.update_output_unit()
