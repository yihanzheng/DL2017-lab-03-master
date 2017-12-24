import torch
import torchvision
import torchvision.transforms as transforms

#load the training and testing data and transform data
class DataLoader(object):

	def __init__(self, data_path, batch_size, n_threads):
		
		self.data_path = data_path
		self.batch_size = batch_size
		self.n_threads = n_threads
	
		#data transform rule
		transform_train = transforms.Compose(
			[transforms.RandomCrop(32, padding=4),
			 transforms.RandomHorizontalFlip(),
			 transforms.ToTensor(),
			 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
		
		transform_test = transforms.Compose(
			[transforms.ToTensor(),
			 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
		
		#take the training and testing data(root,train/test,download,transforms rule)
		trainset = torchvision.datasets.CIFAR10(root = self.data_path,train = True,
							download = True, transform = transform_train)
		testset = torchvision.datasets.CIFAR10(root = self.data_path,train = False,
							download = True, transform = transform_test)
												
		#load the data(data,batch_size,shuffle,threads of data loading)
		self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
								shuffle = True, num_workers = self.n_threads)
		self.test_loader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
								shuffle = False, num_workers = self.n_threads)
													
		self.classes = ('plane', 'car', 'bird', 'cat',
			   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	
	def getloader(self):
        	return self.train_loader, self.test_loader