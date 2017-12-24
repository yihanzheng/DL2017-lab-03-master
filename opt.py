import os
import math
import torch

class Option(object):
	def __init__(self):
	
		#data loader
		self.data_path ="/home/201720145105/DL2017-lab-03-master/data"
		self.batch_size = 128
		self.n_threads = 4
		
		#checkpoint(load exited model/save model)
		self.retrain = None
		self.resume = None
		self.resume_epoch = 0
		self.save_path = "/home/201720145105/model/"
		
		#model
		self.gpu0 = 0
		self.ngpus = 4
		
		#trainer
		self.lr = 0.01
		self.decay_rate = 0.1
		self.ratio = [0.6, 0.8]
		self.momentum = 0.9
		self.weight_decay = 1e-4
		
		#training
		self.nepoch = 150