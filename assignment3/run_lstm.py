# As usual, a bit of setup
import time, os, json
import numpy as np
import matplotlib.pyplot as plt

from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.rnn_layers import *
from cs231n.captioning_solver import CaptioningSolver
from cs231n.classifiers.rnn import CaptioningRNN
from cs231n.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from cs231n.image_utils import image_from_url
import cPickle as pickle

def train_model(data, cell_type='lstm',H=1024, W=512, num_epochs=50, update_rule='adam',
                batch_size=50, lr=1e-3, lr_decay=0.995, 
                verbose=True, print_every=1000):
    captioning_model = CaptioningRNN(
                       cell_type=cell_type,
                       word_to_idx=data['word_to_idx'],
                       input_dim=data['train_features'].shape[1],
                       hidden_dim=H,
                       wordvec_dim=W,
                       dtype=np.float32,
                    )

    captioning_solver = CaptioningSolver(
                        model=captioning_model, 
                        data=data,
                        update_rule=update_rule,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        optim_config={
                          'learning_rate': lr,
                        },
                        lr_decay=lr_decay,
                        verbose=verbose, print_every=print_every,
                      )

    captioning_solver.train()
    return captioning_model, captioning_solver

def get_data(pca=False, max_train=None):
    return load_coco_data(pca_features=pca, max_train=max_train)

if __name__ == "__main__":
	f = "/home/qzz227/projects/applied_research/cs231n/assignment3/cs231n/datasets/models.pkl"
	re_train_model = True
	if re_train_model:
		print "Retraining models"
		max_train = None
		data_pca = get_data(pca=True, max_train=max_train)
		rnn_model_pca, rnn_solver_pca = train_model(data=data_pca, cell_type='rnn', num_epochs=50)
		lstm_model_pca, lstm_solver_pca = train_model(data=data_pca, cell_type='lstm', num_epochs=50)

		
		rnn_model, rnn_solver, lstm_model, lstm_solver = None, None, None, None
	#     data = get_data(pca=False, max_train=max_train)
	#     rnn_model, rnn_solver = train_model(data=data, cell_type='rnn', num_epochs=50)
	#     lstm_model, lstm_solver = train_model(data=data, cell_type='lstm', num_epochs=50)
	else:
		print "Loading models from file"
		with open(f, "rb") as model_file:
			models = pickle.load(model_file)
			rnn_model_pca, rnn_solver_pca = models['rnn_pca']
			lstm_model_pca, lstm_solver_pca = models['lstm_pca']
			rnn_model, rnn_solver = models['rnn']
			lstm_model, lstm_solver = models['lstm']

	rnn_solver_pca, rnn_solver, lstm_solver, lstm_solver_pca = None, None, None, None
	with open(f, "wb") as model_file:
		models = {'rnn_pca' : (rnn_model_pca, rnn_solver_pca),
				  'rnn': (rnn_model, rnn_solver),
				  'lstm_pca' : (lstm_model_pca, lstm_solver_pca),
				  'lstm' : (lstm_model, lstm_solver)}
		pickle.dump(models, model_file, protocol=2)

	captions_file = "/home/qzz227/projects/applied_research/cs231n/assignment3/cs231n/datasets/captions_"
	for split in ['train', 'val']:
		with open(captions_file + split + '.txt', "wb") as c_file:
			if split == 'train':
				minibatch = sample_coco_minibatch(data_pca, split=split, batch_size=60)
			if split == 'val':
				minibatch = sample_coco_minibatch(data_pca, split=split, batch_size=None)
			gt_captions, features, urls = minibatch
			gt_captions = decode_captions(gt_captions, data['idx_to_word'])

			sample_captions_rnn_pca = rnn_model_pca.sample(features)
			sample_captions_rnn_pca = decode_captions(sample_captions_rnn_pca, data['idx_to_word'])

			sample_captions_lstm_pca = lstm_model_pca.sample(features)
			sample_captions_lstm_pca = decode_captions(sample_captions_lstm_pca, data['idx_to_word'])

			for gt_caption, caption_rnn_pca, caption_lstm_pca, url \
			in zip(gt_captions, sample_captions_rnn_pca, sample_captions_lstm_pca, urls):
				c_file.write('%s, RNN_PCA:%s, LSTM_PCA:%s, GT:%s, image:%s' % (split, caption_rnn_pca, caption_lstm_pca, gt_caption, url))
