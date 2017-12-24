import time
#import utils
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from checkpoint import *

class Trainer(object):
	
	def __init__(self, model, opt, optimizer=None):
	
		self.model = model
		self.opt = opt
		#optimizer
		self.criterion = nn.CrossEntropyLoss().cuda()
		optim_list = [{'params': self.model.parameters(), 'lr': self.opt.lr}]
		self.optimizer = optimizer or optim.SGD(optim_list, momentum = self.opt.momentum,
							        weight_decay = self.opt.weight_decay, nesterov=True)
	
	def adjust_learning_rate(self, epoch):
		if (epoch+1.0)/self.opt.nepoch == self.opt.ratio[1] or (epoch+1.0)/self.opt.nepoch==self.opt.ratio[0]:
			print("change the learning rate")
			decay_rate = self.opt.decay_rate
			for param_group in self.optimizer.param_groups:
				param_group['lr'] = param_group['lr'] * decay_rate
				print('learning rate: %.3f' %(param_group['lr']))
			
	def train(self, train_loader, epoch):
		train_loss = 0
		correct = 0
		total = 0
		loss = 0
		accuracy = 0
		
		self.adjust_learning_rate(epoch)
			
		self.model.train()	
		
		for i,data in enumerate(train_loader):
			inputs, labels = data
			inputs, labels = Variable(inputs.cuda()),Variable(labels.cuda())
			outputs = self.model(inputs)
			loss = self.criterion(outputs,labels)
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			
			train_loss += loss.data[0]
			_, predicted = torch.max(outputs.data,1)
			total += labels.size(0)
			correct += predicted.eq(labels.data).cpu().sum()
			print('[%d, %5d] train_loss: %.3f | Acc: %.3f%% (%d/%d)'%
                  		(epoch + 1, i + 1, train_loss/(i+1),100.*correct/total, correct, total))
			out_loss = train_loss / (i+1)
			accuracy = 100.*correct/total
		return out_loss, accuracy
        
	def test(self, test_loader, epoch):
		 test_loss = 0
		 correct = 0
		 total = 0
		 loss = 0
		 accuracy = 0
		 
		 #the model has dropout or batch normalization
		 self.model.eval()
		 
		 for i,data in enumerate(test_loader):
			 inputs, labels = data
			 inputs, labels = Variable(inputs.cuda()),Variable(labels.cuda())
			 outputs = self.model(inputs)
			 loss = self.criterion(outputs,labels)
			
			 test_loss += loss.data[0]
			 _,predicted = torch.max(outputs.data,1)
			 correct += predicted.eq(labels.data).cpu().sum()
			 total += labels.size(0)
			 print('[%d, %5d] test_loss: %.3f | Acc: %.3f%% (%d/%d)'%
                  		(epoch + 1, i + 1, test_loss/(i+1),100.*correct/total, correct, total))
			 out_loss = test_loss/(i+1)
			 accuracy = 100.*correct/total
			
		 return out_loss, accuracy
		
	def class_test():
		class_correct = list(0. for i in range(10))
		class_total = list(0. for i in range(10))
		for data in testloader:
			inputs, labels = data
			outputs = self.model(Variable(inputs).cuda())
			_, predicted = torch.max(outputs.data, 1)
			c = (predicted == labels).cpu().squeeze()
			for i in range(4):
				label = labels[i]
				class_correct[label] += c[i]
				class_total[label] += 1


		for i in range(10):
			print('Accuracy of %5s : %2d %%' % (
				classes[i], 100 * class_correct[i] / class_total[i]))