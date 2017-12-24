import torch
import os


class CheckPoint(object):

	def __init__(self,opt):
		self.retrain = opt.retrain
		self.resume = opt.resume
		self.resume_epoch = opt.resume_epoch
		self.save_path = opt.save_path
		self.check_point_params = {'model':None,
								   'opts':None,
								   'resume_epoch':None
								   }
									
	def retrainmodel(self):
		if os.path.isfile(self.retrain):
			print("|===>Retrain medel from:", self.retrain)
			retrain_data = torch.load(self.retrain)
			self.check_point_params['model'] = retrain_data['model']
			return self.check_point_params
		else:
			assert False, "File is not exited"
	
	def resumemodel(self):
		if os.path.isfile(self.resume):
			print("|===>Resume model from:",self.resume)
			self.check_point_params = torch.load(self.resume)
			if self.resume_epoch != 0:
				self.check_point_params['resume_epoch'] = self.resume_epoch
			return self.check_point_params
		else:
			assert False,"File is not existed"
			
	def save_model(self,epoch=None, model=None, opts=None, best_flag=False):
		if not os.path.isdir(self.save_path):
				os.mkdir(self.save_path)
		
		check_point_params = {'model': model,
							  'opts': opts,
							  'resume_epoch': epoch}
							  
		torch.save(check_point_params, self.save_path+"checkpoint.pkl")
		
		if best_flag:
			# best_model = {'model': utils.list2sequential(model).state_dict()}
			best_model = {'model': model}
			torch.save(best_model, self.save_path+"best_model.pkl")
		