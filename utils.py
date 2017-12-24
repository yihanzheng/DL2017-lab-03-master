import torch.nn as nn
import math
import numpy as np
import torch
import datetime
import matplotlib.pyplot as plt

def dataparallel(model, ngpus, gpu0=0):
    if ngpus == 0:
        assert False, "only support gpu mode"
    gpu_list = list(range(gpu0, gpu0+ngpus))
    assert torch.cuda.device_count() >= gpu0+ngpus, "Invalid Number of GPUs"
    if isinstance(model, list):
        for i in range(len(model)):
            if ngpus >= 2:
                if not isinstance(model[i], nn.DataParallel):
                    model[i] = torch.nn.DataParallel(model[i], gpu_list).cuda()
            else:
                model[i] = model[i].cuda()
    else:
        if ngpus >= 2:
            if not isinstance(model, nn.DataParallel):
                model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:
            model = model.cuda()
    return model
	
def draw_result(result_train = None,result_test =None, result_trainl = None,result_testl =None):
	plt.switch_backend('agg')
	plt.plot(result_train,label="Accuracy_train")
	plt.plot(result_test,label="Accuracy_validation")
	plt.legend()
	plt.xlabel("epoch")
	plt.ylabel("Accuracy")
	plt.title("The result of Cifar10 in VGG16")
    #plt.show()
	plt.savefig('/home/201720145105/result_100.png')
	plt.clf()
	
	plt.plot(result_trainl,label="Loss_train")
	plt.plot(result_testl,label="Loss_validation")
	plt.legend()
	plt.xlabel("epoch")
	plt.ylabel("Loss")
	plt.title("The result of Cifar10 in VGG16")
    #plt.show()
	plt.savefig('/home/201720145105/resultl_100.png')

def writelog(input_data):
	log_file = '/home/201720145105/DL2017-lab-03-master/log.txt'
	txt_file = open(log_file, 'a+')
	txt_file.write(str(input_data) + "\n")
	txt_file.close()
