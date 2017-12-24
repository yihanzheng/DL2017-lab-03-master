from dataloader import *
from model import *
from trainer import *
import time
import os
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import utils
import matplotlib.pyplot as plt
import numpy as np
from opt import *

def main(net_opt = None):
	start_time = time.time()
	
	#get the parameters
	opt = net_opt or Option()
	
	#create dataloader
	data_loader = DataLoader(opt.data_path, opt.batch_size, opt.n_threads)
	train_loader, test_loader = data_loader.getloader()
	print("==>Finish loading data\n")
	
	#define checkpoint and load the model
	check_point = CheckPoint(opt)
	if opt.retrain:
		#model
		check_point_params = check_point.retrainmodel()
		#model,epoch,optimizer
	elif opt.resume:
		check_point_params = check_point.resumemodel()
		#none
	else:
		check_point_params = check_point.check_point_params
	
	#load optimizer
	optimizer = check_point_params['opts']
	
	#load model
	model = check_point_params['model'] or Net()
	model = utils.dataparallel(model=model,ngpus=opt.ngpus,gpu0=opt.gpu0)
	print(model)
	print("==>Finish loading model\n")
	
	start_epoch = check_point_params['resume_epoch'] or 0
	if check_point_params['resume_epoch'] is not None:
		start_epoch += 1
	if start_epoch >= opt.nepoch:
		start_epoch = 1
	
	
	#create trainer
	trainer = Trainer(model=model, opt=opt, optimizer=optimizer)
	
	#training and testing process
	best_loss = 100
	best_acc = 0
	result_train = np.zeros(opt.nepoch)
	result_test = np.zeros(opt.nepoch)
	result_trainl = np.zeros(opt.nepoch)
	result_testl = np.zeros(opt.nepoch)
	for epoch in range(start_epoch,opt.nepoch):
	
		train_loss, train_acc = trainer.train(train_loader=train_loader, epoch=epoch)
		test_loss, test_acc = trainer.test(test_loader=test_loader, epoch=epoch)
		# write and print result
		log_str = "%d\t%.4f\t%.4f\t%.4f\t%.4f\t" % (epoch, train_loss,test_loss, train_acc, test_acc)
		utils.writelog(log_str)
		
		result_train[epoch] = train_acc
		result_test[epoch] = test_acc
		
		result_trainl[epoch] = train_loss
		result_testl[epoch] = test_loss
	
		best_flag = False
		if test_acc>= best_acc:
			best_loss = test_loss
			best_acc = test_acc
			
			best_flag = True
			print("==>Best Result is: Error: %f, Accuracy: %f\n" 
					% (test_loss, test_acc))
		
		check_point.save_model(epoch=epoch, model=trainer.model,
					opts=trainer.optimizer, best_flag=best_flag)
	print("==>Best Result is: Error: %f, Accuracy: %f\n" 
			  % (best_loss, best_acc))
	utils.draw_result(result_train = result_train,result_test =result_test,result_trainl = result_trainl,result_testl =result_testl)
	end_time = time.time()
	time_interval = end_time-start_time
	print("==>Time is: %f\n" 
			  % (time_interval))
			  
	
if __name__ == '__main__':
    main()