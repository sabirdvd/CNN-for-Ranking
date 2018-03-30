#!/usr/bin/env python
#encoding=utf-8

import os
import time
import datetime
import itertools
import numpy as np
import tensorflow as tf
from collections import Counter
from Ranking import Ranking

# Parameters
# ==================================================


def pad_sentences(sentences, sequence_length, padding_word="<PAD/>"):
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence) < sequence_length:
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:sequence_length]
        padded_sentences.append(new_sentence)
    return padded_sentences

def build_vocab(sentences):
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common(300000-1)]
    vocabulary_inv = list(sorted(vocabulary_inv))
    vocabulary_inv.append('<UNK/>')
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_input_data(data_left, data_right, label, vocab):
    vocabset = set(vocab.keys())
    out_left = np.array([[vocab[word] if word in vocabset else vocab['<UNK/>'] for word in sentence ] for sentence in data_left])
    out_right = np.array([[vocab[word] if word in vocabset else vocab['<UNK/>'] for word in sentence ] for sentence in data_right])
    out_y = np.array([[0, 1] if x == 1 else [1, 0] for x in label])
    return [out_left, out_right, out_y]

def load_data(filepath, vocab_tuple=None):
    data = list(set(open(filepath).readlines()))
    data = [d.split(',') for d in data]
    data = filter(lambda x: len(x)==3, data)
    data_left = [x[0].strip().split(' ') for x in data]
    data_right = [x[1].strip().split(' ') for x in data]
    data_label = [int(x[2]) for x in data]
    num_pos = sum(data_label)
    data_left = pad_sentences(data_left, 10)
    data_right = pad_sentences(data_right, 10)
    if vocab_tuple is None:
        vocab, vocab_inv = build_vocab(data_left+data_right)
    else:
        vocab, vocab_inv = vocab_tuple
    data_left, data_right, data_label = build_input_data(data_left, data_right, data_label, vocab)
    '''
    for i in xrange(10):
        print ''.join([vocab_inv[x] for x in data_left[i]])
        print ''.join([vocab_inv[x] for x in data_right[i]])
        print data_label[i]
    '''
    return data_left, data_right, data_label, vocab, vocab_inv, num_pos

def main():
    # Load data
    print("Loading data...")
    x_left_train, x_right_train, y_train, vocab, vocab_inv, num_pos = load_data('data/train.txt')
    x_left_dev, x_right_dev, y_dev, vocab, vocab_inv, num_pos = load_data(('data/test.txt'), (vocab, vocab_inv))

    '''
    # Randomly shuffle data
    np.random.seed(FLAGS.seed)
    shuffle_indices = np.random.permutation(np.arange(len(data_label)))
    data_left = data_left[shuffle_indices]
    data_right = data_right[shuffle_indices]
    data_label = data_label[shuffle_indices]
    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    split_num = int(len(data_label) * FLAGS.eval_split)
    x_left_train, x_left_dev = data_left[:-split_num], data_left[-split_num:]
    x_right_train, x_right_dev = data_right[:-split_num], data_right[-split_num:]
    y_train, y_dev = data_label[:-split_num], data_label[-split_num:]
    print("Vocabulary Size: {:d}".format(len(vocab)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    print("Pos/Neg: {:d}/{:d}".format(num_pos, len(data_label)-num_pos))
    '''

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = Ranking(
                #max_len_left=FLAGS.max_len_left,
                #max_len_left=10,
                max_len_left=10,
                #max_len_right=FLAGS.max_len_right,
                #max_len_right=10,
                max_len_right=10,
                vocab_size=len(vocab),
                embedding_size=64,
                filter_sizes=list(map(int, ('3'))),
                num_filters=64,
                num_hidden=100,
                l2_reg_lambda=0.0)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=20)
            # Initialize all variables
            sess.run(tf.initialize_all_variables())

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join('data/', "runs", timestamp))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            print("Writing to {}\n".format(out_dir))
            checkpoint_prefix = os.path.join(out_dir, "model")

            def batch_iter(all_data, batch_size, num_epochs, shuffle=True):
                data = np.array(all_data)
                data_size = len(data)
                num_batches_per_epoch = int(data_size/batch_size)
                for epoch in range(num_epochs):
                    # Shuffle the data at each epoch
                    if shuffle:
                        shuffle_indices = np.random.permutation(np.arange(data_size))
                        shuffled_data = data[shuffle_indices]
                    else:
                        shuffled_data = data
                    for batch_num in range(num_batches_per_epoch):
                        start_index = batch_num * batch_size
                        end_index = min((batch_num + 1) * batch_size, data_size)
                        yield shuffled_data[start_index:end_index]

            def train_step(x_left_batch, x_right_batch, y_batch):
                feed_dict = {
                cnn.input_left: x_left_batch,
                cnn.input_right: x_right_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 0.5
                }
                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if step % 10 == 0:
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            def dev_step(x_left_batch_dev, x_right_batch_dev, y_batch_dev):
                feed_dict = {
                cnn.input_left: x_left_batch_dev,
                cnn.input_right: x_right_batch_dev,
                cnn.input_y: y_batch_dev,
                cnn.dropout_keep_prob: 1.0
                }
                step, loss, accuracy, sims, pres = sess.run(
                        [global_step, cnn.loss, cnn.accuracy, cnn.sims, cnn.scores],
                        feed_dict)
                '''
                for sims,x1,x2,l,p in zip(sims, x_left_batch_dev, x_right_batch_dev, y_batch_dev, pres):
                    print ''.join([vocab_inv[x] for x in x1])
                    print ''.join([vocab_inv[x] for x in x2])
                    print sims, l, p
                    break
                '''
                return loss,accuracy

            def dev_whole(x_left_dev, x_right_dev, y_dev):
                batches_dev = batch_iter(
                    list(zip(x_left_dev, x_right_dev, y_dev)), 64*2, 1, shuffle=False)
                losses = []
                accuracies = []
                for idx, batch_dev in enumerate(batches_dev):
                    x_left_batch, x_right_batch, y_batch = zip(*batch_dev)
                    loss, accurary = dev_step(x_left_batch, x_right_batch, y_batch)
                    losses.append(loss)
                    accuracies.append(accurary)
                return np.mean(np.array(losses)), np.mean(np.array(accuracies))

            def overfit(dev_loss):
                n = len(dev_loss)
                if n < 5:
                    return False
                for i in xrange(n-4, n):
                    if dev_loss[i] > dev_loss[i-1]:
                        return False
                return True

            # Generate batches
            batches = batch_iter(
                list(zip(x_left_train, x_right_train, y_train)), 64, 500)
            '''
            for batch in batches:
                for x1,x2,y in batch:
                    print ''.join([vocab_inv[x] for x in x1])
                    print ''.join([vocab_inv[x] for x in x2])
                    print y
                break
            '''
            # Training loop. For each batch...
            dev_loss = []
            for batch in batches:
                x1_batch, x2_batch, y_batch = zip(*batch)
                train_step(x1_batch, x2_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % 100 == 0:
                    print("\nEvaluation:")
                    loss, accuracy = dev_whole(x_left_dev, x_right_dev, y_dev)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: dev-aver, loss {:g}, acc {:g}".format(time_str, loss, accuracy))
                    dev_loss.append(accuracy)
                    print("\nRecently accuracy:")
                    print dev_loss[-10:]
                    if overfit(dev_loss):
                        print 'Overfit!!'
                        break
                    print("")
                if current_step % 100 == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

if __name__ == '__main__':
    main()
