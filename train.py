"""
Build and Train Model

Author: avinash.pasupulate@gmail.com
10th May 2020

ToDo:
* Improve syntax
* Write test cases
"""

import os
import re
import yaml
import requests
import numpy as np
import tensorflow as tf
from optparse import OptionParser


def pre_process_audio(url):
    """
    Parsing read file, windows character set not supported in mac.
    Make this more elegant.
    """
    abc_music = requests.get(url).content
    proc_abc_music = re.sub('\%(.*?)\\\\r\\\\n', '', str(abc_music))
    proc_abc_music = re.sub('\#(.*?)\\\\r\\\\n', '', proc_abc_music)
    proc_abc_music = re.sub('I:(.*?)\\\\r\\\\n', '', proc_abc_music)
    proc_abc_music = re.sub('\$(.*?)\\\\r\\\\n', '', proc_abc_music)
    proc_abc_music = re.sub('\\\\r\\\\n', '\n', proc_abc_music)
    proc_abc_music = re.sub('\\\\n', '\n', proc_abc_music)
    proc_abc_music = re.sub("b\\'", '', proc_abc_music)
    proc_abc_music = re.sub("\\\\\\\'", "", proc_abc_music)
    proc_abc_music = re.sub("\n\n\n", "\n\n", proc_abc_music)
    corpus = re.sub('\\\\t', ' ', proc_abc_music)
    return corpus


class music_generation_model(object):
    """
      Model to generate music from training set.
    """

    def __init__(self, config):
        self.config = config

        print("\nDownloading & Pre-processing music in abc notation (text).\n")
        self.corpus = pre_process_audio(self.config['url'])
        # vectorizing vocabulary
        self.vocab = sorted(set(self.corpus))
        # writing vocab to file
        with open('vocabulary.yaml', 'w') as yaml_file:
            yaml.dump(self.vocab, yaml_file, default_flow_style=False)

        self.char2idx = {i: u for u, i in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)

        # vectorized corpus
        self.vectorized_corpus = np.array([self.char2idx[char] for char in self.corpus])

        self.vocab_size = len(self.vocab)
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'model_chk_point')
        return None

    def create_batch(self, vectorized_corpus, segment_len, batch_size):
        """
          Create training batch.
        """
        n = vectorized_corpus.shape[0]-1
        idx = np.random.choice(n-segment_len, batch_size)

        input_batch = [vectorized_corpus[i:i+segment_len] for i in idx]
        output_batch = [vectorized_corpus[i+1:i+segment_len+1] for i in idx]

        x_batch = np.reshape(input_batch, [batch_size, segment_len])
        y_batch = np.reshape(output_batch, [batch_size, segment_len])

        return x_batch, y_batch

    def build_model(self, rnn_units, vocab_size, embeddings_dim, batch_size):
        """
          Creating model layers.
        """
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Embedding(vocab_size, embeddings_dim, batch_input_shape = [batch_size, None]))

        model.add(tf.keras.layers.LSTM(units=rnn_units,
                                      recurrent_activation=self.config['recurrent_activation'],
                                      recurrent_initializer=self.config['recurrent_initializer'],
                                      return_sequences= True,
                                      stateful=True))

        model.add(tf.keras.layers.Dense(vocab_size))
        return model

    def compute_loss(self, labels, logits):
        """
          Compute loss for evaluating model.
        """
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits = True)

    @tf.function
    def train_model(self, x, y):
        with tf.GradientTape() as tape:
            y_hat = self.model(x)
            loss = self.compute_loss(y, y_hat)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def run_train_model(self):
        """
          Train model.
        """
        self.model = self.build_model(self.config['rnn_units'], self.vocab_size, self.config['embeddings_dim'], self.config['batch_size'])
        self.optimizer = tf.keras.optimizers.Adam(float(self.config['learning_rate']))

        for epoch in range(self.config['epochs']):
            x_batch, y_batch = self.create_batch(self.vectorized_corpus, self.config['segment_len'], self.config['batch_size'])
            loss = self.train_model(x_batch, y_batch)
            if epoch % 100 == 0:
                print('Completed epoch {} with loss: {}'.format(epoch, loss.numpy().mean()))
                self.model.save_weights(self.checkpoint_prefix)

        self.model.save_weights(self.checkpoint_prefix)
        return None


def main():
    """
      Preprocessign data, Training, running model and playing generated song.
    """
    parser = OptionParser(usage='usage: python3 train.py config.yaml', version='0.1')
    opts, args = parser.parse_args()

    config = yaml.load(open(args[0], 'r'), Loader=yaml.FullLoader)

    print("\nTraining music generation model.\n")
    musicgen = music_generation_model(config)
    musicgen.run_train_model()


if __name__ == '__main__':
    main()
