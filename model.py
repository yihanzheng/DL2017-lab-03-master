from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nnInit

cfg = {
	'VGG11':[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
	'VGG13':[64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
	'VGG16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
	'VGG19':[64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M'],
}

class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.features = self._make_layers(cfg['VGG16'])
		self.classifier = nn.Linear(512,10)
		self._init_weights()

	def _init_weights(self):
        	for m in self.modules():
            		if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                		nnInit.xavier_normal(m.weight)
                		if m.bias is not None:
                    			m.bias.data.zero_()

	def forward(self,x):
		x = self.features(x)
		x = x.view(x.size(0),-1)
		x = self.classifier(x)
		return x
	
	def _make_layers(self,cfg):
		layers = []
		in_channels = 3
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(2,2)]
			else:
				layers += [nn.Conv2d(in_channels,x,3,padding = 1),
						   nn.BatchNorm2d(x),
						   nn.ReLU(inplace = True)]
				in_channels = x
		layers += [nn.AvgPool2d(kernel_size = 1, stride = 1)]
		return nn.Sequential(*layers)