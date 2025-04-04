import torch
from torch.optim.adam import Adam
import Utils.TimeLogger as logger
from Utils.Log import main_log, olog
from Conf import config
from Model import Model, GaussianDiffusion, Denoise
from DataHandler import DataHandler
import numpy as np
from Utils.Utils import *
import os
import scipy.sparse as sp
import random
import setproctitle
from scipy.sparse import coo_matrix

device = torch.device(f"cuda:{config.base.gpu}" if torch.cuda.is_available() else "cpu")

class Coach:
	def __init__(self, handler: DataHandler):
		self.handler = handler
		
		main_log.info(f"USER: {config.data.user_num}, ITEM: {config.data.item_num}")
		main_log.info(f"NUM OF INTERACTIONS: {self.handler.trainData.__len__()}")

		# init metrics dict: train/test
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = f"Epoch {ep}/{config.train.epoch}, {name}: "
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		main_log.info('Model Prepared')

		recallMax = 0
		ndcgMax = 0
		precisionMax = 0
		bestEpoch = 0

		main_log.info('Model Initialized')

		for ep in range(0, config.train.epoch):
			tstFlag = (ep % config.train.tstEpoch == 0)
			reses = self.trainEpoch()
			olog(self.makePrint('Train', ep, reses, tstFlag))
			if tstFlag:
				reses = self.testEpoch()
				if (reses['Recall'] > recallMax):
					recallMax = reses['Recall']
					ndcgMax = reses['NDCG']
					precisionMax = reses['Precision']
					bestEpoch = ep
				print()
				main_log.info(self.makePrint('Test', ep, reses, tstFlag))
			print()
		print('Best epoch : ', bestEpoch, ' , Recall : ', recallMax, ' , NDCG : ', ndcgMax, ' , Precision', precisionMax)

	def prepareModel(self):
		if config.data.name == 'tiktok':
			self.model = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach(), self.handler.audio_feats.detach()).cuda()
		else:
			self.model = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach()).cuda()

		self.opt = Adam(self.model.parameters(), lr=config.train.lr, weight_decay=0)

		self.diffusion_model = GaussianDiffusion(config.hyper.noise_scale, config.hyper.noise_min, config.hyper.noise_max, config.hyper.steps).cuda()
		
		out_dims = eval(config.base.dims) + [config.data.item_num]
		in_dims = out_dims[::-1]
		self.denoise_model_image = Denoise(in_dims, out_dims, config.base.d_emb_size, norm=config.train.norm).cuda()
		self.denoise_opt_image = Adam(self.denoise_model_image.parameters(), lr=config.train.lr, weight_decay=0)

		out_dims = eval(config.base.dims) + [config.data.item_num]
		in_dims = out_dims[::-1]
		self.denoise_model_text = Denoise(in_dims, out_dims, config.base.d_emb_size, norm=config.train.norm).cuda()
		self.denoise_opt_text = Adam(self.denoise_model_text.parameters(), lr=config.train.lr, weight_decay=0)

		if config.data.name == 'tiktok':
			out_dims = eval(config.base.dims) + [config.data.item_num]
			in_dims = out_dims[::-1]
			self.denoise_model_audio = Denoise(in_dims, out_dims, config.base.d_emb_size, norm=config.train.norm).cuda()
			self.denoise_opt_audio = Adam(self.denoise_model_audio.parameters(), lr=config.train.lr, weight_decay=0)

	def normalizeAdj(self, mat): 
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def buildUIMatrix(self, u_list, i_list, edge_list):
		mat = coo_matrix((edge_list, (u_list, i_list)), shape=(config.data.user_num, config.data.item_num), dtype=np.float32)

		a = sp.csr_matrix((config.data.user_num, config.data.user_num))
		b = sp.csr_matrix((config.data.item_num, config.data.item_num))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)

		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)

		return torch.sparse_coo_tensor(idxs, vals, shape, device=device)

	def trainEpoch(self):
		trnLoader = self.handler.trainLoader
		trnLoader.dataset.negSampling()
		epLoss, epRecLoss, epClLoss = 0, 0, 0
		epDiLoss = 0
		epDiLoss_image, epDiLoss_text = 0, 0
		if config.data.name == 'tiktok':
			epDiLoss_audio = 0
		steps = trnLoader.dataset.__len__() // config.train.batch

		diffusionLoader = self.handler.diffusionLoader

		for i, batch in enumerate(diffusionLoader):
			batch_item, batch_index = batch
			batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

			iEmbeds = self.model.getItemEmbeds().detach()
			uEmbeds = self.model.getUserEmbeds().detach()

			image_feats = self.model.getImageFeats().detach()
			text_feats = self.model.getTextFeats().detach()
			if config.data.name == 'tiktok':
				audio_feats = self.model.getAudioFeats().detach()

			self.denoise_opt_image.zero_grad()
			self.denoise_opt_text.zero_grad()
			if config.data.name == 'tiktok':
				self.denoise_opt_audio.zero_grad()

			diff_loss_image, gc_loss_image = self.diffusion_model.training_losses(self.denoise_model_image, batch_item, iEmbeds, batch_index, image_feats)
			diff_loss_text, gc_loss_text = self.diffusion_model.training_losses(self.denoise_model_text, batch_item, iEmbeds, batch_index, text_feats)
			if config.data.name == 'tiktok':
				diff_loss_audio, gc_loss_audio = self.diffusion_model.training_losses(self.denoise_model_audio, batch_item, iEmbeds, batch_index, audio_feats)

			loss_image = diff_loss_image.mean() + gc_loss_image.mean() * config.hyper.e_loss
			loss_text = diff_loss_text.mean() + gc_loss_text.mean() * config.hyper.e_loss
			if config.data.name == 'tiktok':
				loss_audio = diff_loss_audio.mean() + gc_loss_audio.mean() * config.hyper.e_loss

			epDiLoss_image += loss_image.item()
			epDiLoss_text += loss_text.item()
			if config.data.name == 'tiktok':
				epDiLoss_audio += loss_audio.item()

			if config.data.name == 'tiktok':
				loss = loss_image + loss_text + loss_audio
			else:
				loss = loss_image + loss_text

			loss.backward()

			self.denoise_opt_image.step()
			self.denoise_opt_text.step()
			if config.data.name == 'tiktok':
				self.denoise_opt_audio.step()

			olog(f"Diffusion Step {i}/{diffusionLoader.dataset.__len__() // config.train.batch}")

		print()
		main_log.info('Start to re-build UI matrix')

		with torch.no_grad():

			u_list_image = []
			i_list_image = []
			edge_list_image = []

			u_list_text = []
			i_list_text = []
			edge_list_text = []

			if config.data.name == 'tiktok':
				u_list_audio = []
				i_list_audio = []
				edge_list_audio = []

			for _, batch in enumerate(diffusionLoader):
				batch_item, batch_index = batch
				batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

				# image
				denoised_batch = self.diffusion_model.p_sample(self.denoise_model_image, batch_item, config.hyper.sampling_steps, config.train.sampling_noise)
				top_item, indices_ = torch.topk(denoised_batch, k=config.hyper.rebuild_k)

				for i in range(batch_index.shape[0]):
					for j in range(indices_[i].shape[0]): 
						u_list_image.append(int(batch_index[i].cpu().numpy()))
						i_list_image.append(int(indices_[i][j].cpu().numpy()))
						edge_list_image.append(1.0)

				# text
				denoised_batch = self.diffusion_model.p_sample(self.denoise_model_text, batch_item, config.hyper.sampling_steps, config.train.sampling_noise)
				top_item, indices_ = torch.topk(denoised_batch, k=config.hyper.rebuild_k)

				for i in range(batch_index.shape[0]):
					for j in range(indices_[i].shape[0]): 
						u_list_text.append(int(batch_index[i].cpu().numpy()))
						i_list_text.append(int(indices_[i][j].cpu().numpy()))
						edge_list_text.append(1.0)

				if config.data.name == 'tiktok':
					# audio
					denoised_batch = self.diffusion_model.p_sample(self.denoise_model_audio, batch_item, config.hyper.sampling_steps, config.train.sampling_noise)
					top_item, indices_ = torch.topk(denoised_batch, k=config.hyper.rebuild_k)

					for i in range(batch_index.shape[0]):
						for j in range(indices_[i].shape[0]): 
							u_list_audio.append(int(batch_index[i].cpu().numpy()))
							i_list_audio.append(int(indices_[i][j].cpu().numpy()))
							edge_list_audio.append(1.0)

			# image
			u_list_image = np.array(u_list_image)
			i_list_image = np.array(i_list_image)
			edge_list_image = np.array(edge_list_image)
			self.image_UI_matrix = self.buildUIMatrix(u_list_image, i_list_image, edge_list_image)
			self.image_UI_matrix = self.model.edgeDropper(self.image_UI_matrix)

			# text
			u_list_text = np.array(u_list_text)
			i_list_text = np.array(i_list_text)
			edge_list_text = np.array(edge_list_text)
			self.text_UI_matrix = self.buildUIMatrix(u_list_text, i_list_text, edge_list_text)
			self.text_UI_matrix = self.model.edgeDropper(self.text_UI_matrix)

			if config.data.name == 'tiktok':
				# audio
				u_list_audio = np.array(u_list_audio)
				i_list_audio = np.array(i_list_audio)
				edge_list_audio = np.array(edge_list_audio)
				self.audio_UI_matrix = self.buildUIMatrix(u_list_audio, i_list_audio, edge_list_audio)
				self.audio_UI_matrix = self.model.edgeDropper(self.audio_UI_matrix)

		main_log.info('UI matrix built!')

		for i, tem in enumerate(trnLoader):
			ancs, poss, negs = tem
			ancs = ancs.long().cuda()
			poss = poss.long().cuda()
			negs = negs.long().cuda()

			self.opt.zero_grad()

			if config.data.name == 'tiktok':
				usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
			else:
				usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)
			ancEmbeds = usrEmbeds[ancs]
			posEmbeds = itmEmbeds[poss]
			negEmbeds = itmEmbeds[negs]
			scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
			bprLoss = - (scoreDiff).sigmoid().log().sum() / config.train.batch
			regLoss = self.model.reg_loss() * config.train.reg
			loss = bprLoss + regLoss
			
			epRecLoss += bprLoss.item()
			epLoss += loss.item()

			if config.data.name == 'tiktok':
				usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2, usrEmbeds3, itmEmbeds3 = self.model.forward_cl_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
			else:
				usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2 = self.model.forward_cl_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)
			if config.data.name == 'tiktok':
				clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, config.hyper.temp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, config.hyper.temp)) * config.hyper.ssl_reg
				clLoss += (contrastLoss(usrEmbeds1, usrEmbeds3, ancs, config.hyper.temp) + contrastLoss(itmEmbeds1, itmEmbeds3, poss, config.hyper.temp)) * config.hyper.ssl_reg
				clLoss += (contrastLoss(usrEmbeds2, usrEmbeds3, ancs, config.hyper.temp) + contrastLoss(itmEmbeds2, itmEmbeds3, poss, config.hyper.temp)) * config.hyper.ssl_reg
			else:
				clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, config.hyper.temp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, config.hyper.temp)) * config.hyper.ssl_reg

			clLoss1 = (contrastLoss(usrEmbeds, usrEmbeds1, ancs, config.hyper.temp) + contrastLoss(itmEmbeds, itmEmbeds1, poss, config.hyper.temp)) * config.hyper.ssl_reg
			clLoss2 = (contrastLoss(usrEmbeds, usrEmbeds2, ancs, config.hyper.temp) + contrastLoss(itmEmbeds, itmEmbeds2, poss, config.hyper.temp)) * config.hyper.ssl_reg
			if config.data.name == 'tiktok':
				clLoss3 = (contrastLoss(usrEmbeds, usrEmbeds3, ancs, config.hyper.temp) + contrastLoss(itmEmbeds, itmEmbeds3, poss, config.hyper.temp)) * config.hyper.ssl_reg
				clLoss_ = clLoss1 + clLoss2 + clLoss3
			else:
				clLoss_ = clLoss1 + clLoss2

			if config.base.cl_method == 1:
				clLoss = clLoss_

			loss += clLoss

			epClLoss += clLoss.item()

			loss.backward()
			self.opt.step()

			olog(f"Step {i}/{steps}: bpr : {bprLoss.item():.3f} ; reg : {regLoss.item():.3f} ; cl : {clLoss.item():.3f}")
		print()

		ret = dict()
		ret['Loss'] = epLoss / steps
		ret['BPR Loss'] = epRecLoss / steps
		ret['CL loss'] = epClLoss / steps
		ret['Di image loss'] = epDiLoss_image / (diffusionLoader.dataset.__len__() // config.train.batch)
		ret['Di text loss'] = epDiLoss_text / (diffusionLoader.dataset.__len__() // config.train.batch)
		if config.data.name == 'tiktok':
			ret['Di audio loss'] = epDiLoss_audio / (diffusionLoader.dataset.__len__() // config.train.batch)
		return ret

	def testEpoch(self):
		testData = self.handler.testData
		tstLoader = self.handler.testLoader
		epRecall, epNdcg, epPrecision = [0] * 3
		i = 0
		num = testData.__len__()
		steps = num // config.train.tstBat

		if config.data.name == 'tiktok':
			usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
		else:
			usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)

		for usr, trnMask in tstLoader:
			i += 1
			usr = usr.long().cuda()
			trnMask = trnMask.cuda()
			allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
			_, topLocs = torch.topk(allPreds, config.base.topk)
			recall, ndcg, precision = self.calcRes(topLocs.cpu().numpy(), testData.test_use_its, usr)
			epRecall += recall
			epNdcg += ndcg
			epPrecision += precision
			olog(f"Step {i}/{steps}: recall = {recall:.2f}, ndcg = {ndcg:.2f} , precision = {precision:.2f}")
		ret = dict()
		ret['Recall'] = epRecall / num
		ret['NDCG'] = epNdcg / num
		ret['Precision'] = epPrecision / num
		return ret

	def calcRes(self, topLocs, test_u_its, batIds):
		assert topLocs.shape[0] == len(batIds)
		allRecall = allNdcg = allPrecision = 0
		for i in range(len(batIds)):
			temTopLocs = list(topLocs[i])
			tmp_u_its = test_u_its[batIds[i]]
			tstNum = len(tmp_u_its)
			maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, config.base.topk))])
			recall = dcg = precision = 0
			for val in tmp_u_its:
				if val in temTopLocs:
					recall += 1
					dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
					precision += 1
			recall = recall / tstNum
			ndcg = dcg / maxDcg
			precision = precision / config.base.topk
			allRecall += recall
			allNdcg += ndcg
			allPrecision += precision
		return allRecall, allNdcg, allPrecision

def seed_it(seed):
	"""
	Set the random seed for all
	"""
	random.seed(seed)
	os.environ["PYTHONSEED"] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	#torch.cuda.manual_seed(seed)  # if device is not cuda, this is ignored
	#torch.cuda.manual_seed_all(seed)
	#? will deterministic conflict with benchmark?
	torch.backends.cudnn.deterministic = True  # force cudnn to be deterministic
	torch.backends.cudnn.benchmark = True  # enable cudnn auto-tuner to find the optimal set of algorithms for the hardware
	torch.backends.cudnn.enabled = True
	

if __name__ == '__main__':
	seed_it(config.base.seed)

	logger.saveDefault = True  #todo: reconstruct log module
	
	main_log.info('Start')
	data_handler = DataHandler()
	data_handler.LoadData()
	main_log.info('Load Data')

	coach = Coach(data_handler)
	coach.run()