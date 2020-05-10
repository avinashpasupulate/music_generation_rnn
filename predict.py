"""
Generate music with model

Author: avinash.pasupulate@gmail.com
10th May 2020

ToDo:
* Improve syntax
* Write test cases
"""

import os
import yaml
import numpy as np
import tensorflow as tf
from datetime import datetime as dt
from optparse import OptionParser

from train import music_generation_model

# install dependencies
os.system('sudo chmod 755 dependencies.sh & ./dependencies.sh')


class predict_next_note(object):
    """
        Generate Music.
    """

    def __init__(self, config):
        self.config = config
        self.vocab = yaml.load(open('./vocabulary.yaml', 'r'), Loader=yaml.FullLoader)
        # write to and load vocabulary as dict from yaml file
        self.char2idx = {i: u for u, i in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)

        self.vocab_size = len(self.vocab)
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'model_chk_point')
        return None

    def load_trained_model(self):
        """
          Load trained model weights.
        """
        model = music_generation_model(self.config).build_model(self.config['rnn_units'], self.vocab_size, self.config['embeddings_dim'], batch_size = 1)

        model.load_weights(tf.train.latest_checkpoint(self.checkpoint_dir))
        model.build(tf.TensorShape([1, None]))
        model.summary()
        return model

    def generate_text(self, model, start_string, generation_length=1000):
        input_eval = [self.char2idx[char] for char in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        generated_text = []
        model.reset_states()

        for i in range(generation_length):
            predictions = model(input_eval)
            predictions = tf.squeeze(predictions, 0)
            prediction_id = tf.random.categorical(predictions, num_samples = 1)[-1, 0].numpy()

            # note the variable name prediction for this iteration (aka time step) becomes the input for the next.
            input_eval = tf.expand_dims([prediction_id], 0)

            generated_text.append(self.idx2char[prediction_id])
        return (start_string+''.join(generated_text))

    @staticmethod
    def write_audio(sample_music, base_path, filename):
        """
        Converts abc to midi and midi to wav. Then provides a player on IPython notebook.
        """
        os.chdir(base_path+'/generated_music')
        with open(filename+'.abc', 'w') as f:
            f.write(sample_music)
        cmd = 'abc2midi {filename}.abc -o "{filename}.mid" & timidity "{filename}.mid" -Ow "{filename}.wav"'.format(base_path = base_path, filename = filename)
        ret = os.system(cmd)
        if ret == 0:
            print("WAV file generated.")
        else:
            print("Unable to play file.")


def main():
    parser = OptionParser(usage='usage: python3 predict.py config.yaml', version='0.1')
    opts, args = parser.parse_args()

    config = yaml.load(open(args[0], 'r'), Loader=yaml.FullLoader)
    print("\nLoading model weights to generate music data.\n")
    predict = predict_next_note(config)
    model = predict.load_trained_model()

    print("\nConverting generated music to wav.\n")
    generated_music = ['X:'+i.strip() for i in predict.generate_text(model, 'X:', generation_length = 1000).split('X:') if len(i) > 10]
    for u, i in enumerate(generated_music):
        try:
            predict.write_audio(i, config['base_path'], 'generated_song_{u}_{date}'.format(u=u, date=dt.strftime(dt.today(), '%d%m%Y')))
            print('Writing Song: {} Complete.'.format(u))
        except:
            pass


if __name__ == '__main__':
    main()
