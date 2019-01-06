import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
import pickle as cPickle
from nltk.translate.bleu_score import corpus_bleu
import yaml
import shutil
import postprocessing as POST
from six.moves import xrange

with open("SeqGAN.yaml") as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

os.environ['CUDA_VISIBLE_DEVICES'] = config['GPU'] ##最大问题，感觉GPU没有用上
#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = config['EMB_DIM'] # embedding dimension
HIDDEN_DIM = config['HIDDEN_DIM'] # hidden state dimension of lstm cell
SEQ_LENGTH = config['SEQ_LENGTH'] # sequence length
START_TOKEN = config['START_TOKEN']
PRE_GEN_EPOCH = config['PRE_GEN_EPOCH'] # supervise (maximum likelihood estimation) epochs for generator
PRE_DIS_EPOCH = config['PRE_DIS_EPOCH'] # supervise (maximum likelihood estimation) epochs for discriminator
SEED = config['SEED']
BATCH_SIZE = config['BATCH_SIZE']

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = config['dis_embedding_dim']
dis_filter_sizes = config['dis_filter_sizes']
dis_num_filters = config['dis_num_filters']
dis_dropout_keep_prob = config['dis_dropout_keep_prob']
dis_l2_reg_lambda = config['dis_l2_reg_lambda']
dis_batch_size = config['dis_batch_size']

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = config['TOTAL_BATCH']
# vocab size for our custom data
vocab_size = config['vocab_size']
# positive data, containing real music sequences
positive_file = config['datapath'] + config['positive_file']
# negative data from the generator, containing fake sequences
negative_file = config['datapath'] + config['negative_file']
valid_file = config['datapath'] + config['valid_file']
generated_num = config['generated_num']

epochs_generator = config['epochs_generator']
epochs_discriminator = config['epochs_discriminator']

path = config['datapath']

#生成样本？
def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # unconditinally generate random samples 无条件生成随机样本
    # it is used for test sample generation & negative data generation 这是被用于 测试样本生成&负例样本生成
    # called per D learning phase 被调用在每一个D的学习周期

    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))
    # dump the pickle data
    with open(output_file, 'wb') as fp:
        cPickle.dump(generated_samples, fp, protocol=2)

#预训练 周期
def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch 利用 MLE(?)进行生成器的预训练
    # independent of D, the standard RNN learning 独立与D，是标准的RNN学习， 玛德和我想的一样！！！ 是不是和我的一样？
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)#返回LOSS

# new implementations 新的实现
def calculate_train_loss_epoch(sess, trainableav_model, data_loader): 
    # calculate the train loss for the generator  计算 生成器的 训练的 损失
    # same for pre_train_epoch, but without the supervised grad update 和 pre train epoch一样，只是没有了 监督的梯度更新（？）
    # used for observing overfitting and stability of the generator 用于 观察 生成器的 过拟合以及稳定性的 情况
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch): 
        batch = data_loader.next_batch()
        # note the newly implementated method call for the model 注意使用了新的 实现的 方法 去调用
        # calculate_nll_loss_step calculate the node up to g_loss, but does not calculate the update node 那个函数计算了g_loss上的node，但是没有计算更新的node
        g_loss = trainable_model.calculate_nll_loss_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)

# 这个其实没怎么明白 BLEU实质是对两个句子的共现词频率计算 那应该就是判断两个相似的拉，，对是的！ 我要改的就是这里！！
def calculate_bleu(sess, trainable_model, data_loader):
    # bleu score implementationa BLEU实质是对两个句子的共现词频率计算 这里是他的实现
    # used for performance evaluation for pre-training & adv. training 用于 评估 预训练和 对抗训练的 性能
    # separate true dataset to the valid set 分割真的数据集到 验证集
    # conditionally generate samples from the start token of the valid set 从valid集合的 开始token下 条件生成样本
    # measure similarity with nltk corpus BLEU

    data_loader.reset_pointer()
    bleu_avg = 0
    #具体之后再看，这块是重点！ 重中之重，因为之后我切入点就是这块，我的切入点。
    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        # predict from the batch
        start_tokens = batch[:, 0]
        prediction = trainable_model.predict(sess, batch, start_tokens)
        # argmax to convert to vocab
        prediction = np.argmax(prediction, axis=2)

        # cast batch and prediction to 2d list of strings
        batch_list = batch.astype(np.str).tolist()
        pred_list = prediction.astype(np.str).tolist()

        bleu = 0
        # calculate bleu for each sequence
        for i in range(len(batch_list)):
            bleu += corpus_bleu(batch_list[i], pred_list[i])
        bleu = bleu / len(batch_list)
        bleu_avg += bleu
    bleu_avg = bleu_avg / data_loader.num_batch

    return bleu_avg

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    # data loaders declaration
    # loaders for generator, discriminator, and additional validation data loader
    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    dis_data_loader = Dis_dataloader(BATCH_SIZE)
    eval_data_loader = Gen_Data_loader(BATCH_SIZE)

    # define generator and discriminator
    # general structures are same with the original model
    # learning rates for generator needs heavy tuning for general use
    # l2 reg for D & G also affects performance
    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim,
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    # VRAM limitation for efficient deployment
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())
    # define saver
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
    # generate real data from the true dataset
    gen_data_loader.create_batches(positive_file)
    # generate real validation data from true validation dataset
    eval_data_loader.create_batches(valid_file)

    log = open( path + '/save/experiment-log.txt', 'w')
    if config['pretrain'] == True:
        #  pre-train generator
        print('Start pre-training...')
        log.write('pre-training...\n')
        for epoch in xrange(PRE_GEN_EPOCH):
            # calculate the loss by running an epoch
            loss = pre_train_epoch(sess, generator, gen_data_loader)

            # measure bleu score with the validation set
            bleu_score = calculate_bleu(sess, generator, eval_data_loader)
            # since the real data is the true data distribution, only evaluate the pretraining loss
            # note the absence of the oracle model which is meaningless for general use
            buffer = 'pre-train epoch: ' + str(epoch) + ' pretrain_loss: ' + str(loss) + ' bleu: ' + str(bleu_score)
            print(buffer)
            log.write(buffer)

            # generate 5 test samples per epoch
            # it automatically samples from the generator and postprocess to midi file
            # midi files are saved to the pre-defined folder
            if epoch == 0:
                generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
                POST.main(negative_file, 5, -1)
            elif epoch == PRE_GEN_EPOCH - 1:
                generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
                POST.main(negative_file, 5, -PRE_GEN_EPOCH)


        print('Start pre-training discriminator...')
        # Train 3 epoch on the generated data and do this for 50 times
        # this trick is also in spirit of the original work, but the epoch strategy needs tuning
        for epochs in range(PRE_DIS_EPOCH):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)
            D_loss = 0
            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in xrange(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _ = sess.run(discriminator.train_op, feed)
                    D_loss += discriminator.loss.eval(feed, session=sess) ##这里 loss.eval到底指的是啥
                    #你可以使用sess.run()在同一步获取多个tensor中的值，使用Tensor.eval()时只能在同一步当中获取一个tensor值，
                    #并且每次使用 eval 和 run时，都会执行整个计算图
            buffer = 'epoch: ' + str(epochs+1) + '  D loss: ' + str(D_loss/dis_data_loader.num_batch/3)
            print(buffer)
            log.write(buffer)

        # save the pre-trained checkpoint for future use
        # if one wants adv. training only, comment out the pre-training section after the save
        save_checkpoint(sess, saver,PRE_GEN_EPOCH, PRE_DIS_EPOCH)

    # define rollout target object
    # the second parameter specifies target update rate
    # the higher rate makes rollout "conservative", with less update from the learned generator
    # we found that higher update rate stabilized learning, constraining divergence of the generator
    rollout = ROLLOUT(generator, 0.9)

    print('#########################################################################')
    print('Start Adversarial Training...')
    log.write('adversarial training...\n')
    if config['pretrain'] == False:
        # load checkpoint of pre-trained model
        load_checkpoint(sess, saver)
    for total_batch in range(TOTAL_BATCH):
        G_loss = 0
        # Train the generator for one step
        for it in range(epochs_generator):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, config['rollout_num'], discriminator)
            feed = {generator.x: samples, generator.rewards: rewards}
            _ = sess.run(generator.g_updates, feed_dict=feed)
            G_loss += generator.g_loss.eval(feed, session=sess)

        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator
        D_loss = 0
        for _ in range(epochs_discriminator):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)
            for _ in range(3):
                dis_data_loader.reset_pointer()

                for it in xrange(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _ = sess.run(discriminator.train_op, feed)
                    D_loss += discriminator.loss.eval(feed, session=sess)

        # measure stability and performance evaluation with bleu score
        buffer = 'epoch: ' + str(total_batch+1) + \
                 ',  G_adv_loss: %.12f' % (G_loss/epochs_generator) + \
                 ',  D loss: %.12f' % (D_loss/epochs_discriminator/3) + \
                 ',  bleu score: %.12f' % calculate_bleu(sess, generator, eval_data_loader)
        #在这里，bleu有没有用到呢，有没有作用在D的训练中
        print(buffer)
        log.write(buffer)

        # generate random test samples and postprocess the sequence to midi file
        generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file + "_EP_" + str(total_batch))
        POST.main(negative_file + "_EP_" + str(total_batch), 5, total_batch)
    log.close()


# methods for loading and saving checkpoints of the model
def load_checkpoint(sess, saver):
    #ckpt = tf.train.get_checkpoint_state('save')
    #if ckpt and ckpt.model_checkpoint_path:
    #saver.restore(sess, tf.train.latest_checkpoint('save'))
    ckpt = 'pretrain_g'+str(config['PRE_GEN_EPOCH'])+'_d'+str(config['PRE_DIS_EPOCH'])+'.ckpt'
    saver.restore(sess, path + '/save/' + ckpt)
    print('checkpoint {} loaded'.format(ckpt))
    return


def save_checkpoint(sess, saver, g_ep, d_ep):
    checkpoint_path = os.path.join('save', 'pretrain_g'+str(g_ep)+'_d'+str(d_ep)+'.ckpt')
    saver.save(sess, checkpoint_path)
    print("model saved to {}".format(checkpoint_path))
    return

if __name__ == '__main__':
    main()
