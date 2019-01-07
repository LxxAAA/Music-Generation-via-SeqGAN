#数据读取 生成器 但是没有和midi相关的代码，可能只是生成batch？ 
import numpy as np
import pickle as cPickle
import yaml
import random

with open("SeqGAN.yaml") as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

class Gen_Data_loader(): #主要是操作batch
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.token_stream = []

    def create_batches(self, data_file): #创建一个batch
        self.token_stream = []
        with open(data_file, 'rb') as f: #读取数据
            # load pickle data
            data = cPickle.load(f)
            for line in data: #
                parse_line = [int(x) for x in line]
                if len(parse_line) == config['SEQ_LENGTH']: #这块是做筛选么，能加assert么
                    self.token_stream.append(parse_line)

        self.num_batch = int(len(self.token_stream) / self.batch_size) #batch总数
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size] #从0到 batch数目*batch大小，这是固定batch让它对齐吧
        #np.split是分割，split(A,B,C),A是要分割的数组，B是分为多少份，C是维度，0列或者1行
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0) #？？得到的是最后完成的数组，怎么得到的不是十分清楚
        self.pointer = 0#应该是指针把

    def next_batch(self): #下一个batch， 直接把指针后移了
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self): #指针复位
        self.pointer = 0
        # shuffle the data
        np.random.shuffle(self.sequence_batch)#batch重新打乱


class Dis_dataloader(): #判别器 的 数据读取
    def __init__(self, batch_size):
        self.batch_size = batch_size #
        self.sentences = np.array([])
        self.labels = np.array([])

    def load_train_data(self, positive_file, negative_file):
        # Load data
        positive_examples = [] #正例
        negative_examples = [] #负例
        with open(positive_file, 'rb')as fin: #
            data = cPickle.load(fin) #
            for line in data:
                parse_line = [int(x) for x in line]
                if len(parse_line) != config['SEQ_LENGTH']:
                    continue
                if len(parse_line) == config['SEQ_LENGTH']:
                    positive_examples.append(parse_line)
        with open(negative_file, 'rb')as fin:
            data = cPickle.load(fin)
            for line in data:
                parse_line = [int(x) for x in line]
                if len(parse_line) != config['SEQ_LENGTH']:
                    continue
                if len(parse_line) == config['SEQ_LENGTH']:
                    negative_examples.append(parse_line)

        #上面两个读取了 正例 和 负例 的样本
        # pos / neg only batches implementation 
        # shuffle the pos and neg examples
        random.shuffle(positive_examples)
        random.shuffle(negative_examples)

        # ditch the pos & neg samples not matching the batch size 把多的部分扔了
        if len(positive_examples) % self.batch_size != 0:
            positive_examples = positive_examples[:-(len(positive_examples) % self.batch_size)]
        if len(negative_examples) % self.batch_size != 0:
            negative_examples = negative_examples[:-(len(negative_examples) % self.batch_size)]

        self.sentences = np.array(positive_examples + negative_examples)

        # Generate labels 制作标签
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]

        self.labels = np.concatenate([positive_labels, negative_labels], 0)



        # shuffling here mixes positive & negative data
        # however, separating pos & neg batches is said to be better
        """
        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]
        """
        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0


    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

        # shuffle the data batch-wise when reset
        shuffle_temp = list(zip(self.sentences_batches, self.labels_batches))
        np.random.shuffle(shuffle_temp)
        self.sentences_batches, self.labels_batches = zip(*shuffle_temp)
