import torch
from torch import Tensor
from torch.optim.adam import Adam
from Utils.Log import Log
from Conf import load_config, Config
from Model import Model, GaussianDiffusion, Denoise
from DataHandler import DataHandler, DiffusionData
import numpy as np
from Utils.Utils import *
import os
import random
from scipy.sparse import coo_matrix, csr_matrix
import ast
import argparse

class Coach:
	def __init__(self, handler: DataHandler, config: Config):
		self.handler = handler
		self.config = config
		self.device = torch.device(f"cuda:{self.config.base.gpu}" if torch.cuda.is_available() else "cpu")
		main_log.info(f"USER: {self.config.data.user_num}, ITEM: {self.config.data.item_num}")
		main_log.info(f"NUM OF INTERACTIONS: {len(self.handler.trainData)}")

	def makePrint(self, name, epoch, results: dict):
		"""Splicing model metric dict into strings"""
		result_str = f"Epoch {epoch}/{self.config.train.epoch}, {name}: "
		for metric in results:
			val = results[metric]
			result_str += f"{metric}={val:.4f}, "
		result_str = result_str[:-2] + '  ' #? del blank and `:`?
		return result_str
	
	def need_update(self, l1: list, l2: list) -> bool:
		"""
		If the number of elements in l1 that are greater than l2 is strictly more than half (rounded up), return True.
		"""
		if len(l1) != len(l2):
			raise ValueError("Result list shape should be the same!")

		count_l1_greater = sum(1 for a, b in zip(l1, l2) if a > b)
		half_length = (len(l1) + 1) // 2
		return count_l1_greater >= half_length

	def run(self):
		self.prepareModel()
		main_log.info('Model Initialized ✅')

		his_recall_max, recallMax = 0, 0
		his_ndcg_max, ndcgMax = 0, 0
		his_prec_max, precisionMax = 0, 0
		bestEpoch = 0  #todo: early stop

		main_log.info('Start training 🚀')
		for epoch in range(0, self.config.train.epoch):
			tstFlag = (epoch % self.config.train.tstEpoch == 0) #? always be True?
			result = self.trainEpoch()
			main_log.info(self.makePrint('⏩ Train', epoch, result))
			if tstFlag:
				result = self.testEpoch()
				re_list = [result['Recall'], result['NDCG'], result['Precision']]
				his_recall_max = max(his_recall_max, result['Recall'])
				his_ndcg_max = max(his_ndcg_max, result['NDCG'])
				his_prec_max = max(his_prec_max, result['Precision'])
				if (self.need_update(re_list, [recallMax, ndcgMax, precisionMax])):
					recallMax = result['Recall']
					ndcgMax = result['NDCG']
					precisionMax = result['Precision']
					bestEpoch = epoch
				main_log.info(self.makePrint('🧪 Test', epoch, result))
			main_log.info(f"💡 Current best: Epoch: {bestEpoch}, Recall: {recallMax:.4f}, NDCG: {ndcgMax:.4f}, Precision: {precisionMax:.4f}")
		main_log.info(f"Best epoch: {bestEpoch}, Recall: {recallMax:.5f}, NDCG: {ndcgMax:.5f}, Precision: {precisionMax:.5f}")
		main_log.info(f"Historical best: Recall: {his_recall_max:.5f}, NDCG: {his_ndcg_max:.5f}, Precision: {his_prec_max:.5f}")

	def prepareModel(self):
		"""Init DiffMM, Diffusion, Denoise Models"""
		if self.config.data.name == 'tiktok':
			self.model = Model(self.config, self.handler.image_feats.detach(), self.handler.text_feats.detach(), self.handler.audio_feats.detach()).cuda(self.device)
		else:
			self.model = Model(self.config, self.handler.image_feats.detach(), self.handler.text_feats.detach()).cuda(self.device)

		self.opt = Adam(self.model.parameters(), lr=self.config.train.lr, weight_decay=0)

		self.diffusion_model = GaussianDiffusion(self.config).cuda(self.device)
		
		out_dims = ast.literal_eval(self.config.base.denoise_dim) + [self.config.data.item_num] # [denoise_dim, item_num]
		in_dims = out_dims[::-1]  # [item_num, denoise_dim]
		self.image_denoise_model = Denoise(in_dims, out_dims, config).cuda(self.device)
		self.image_denoise_opt = Adam(self.image_denoise_model.parameters(), lr=self.config.train.lr, weight_decay=0)

		self.text_denoise_model = Denoise(in_dims, out_dims, config).cuda(self.device)
		self.text_denoise_opt = Adam(self.text_denoise_model.parameters(), lr=self.config.train.lr, weight_decay=0)

		if self.config.data.name == 'tiktok':
			self.audio_denoise_model = Denoise(in_dims, out_dims, config).cuda(self.device)
			self.audio_denoise_opt = Adam(self.audio_denoise_model.parameters(), lr=self.config.train.lr, weight_decay=0)

	def makeTorchAdj(self, u_list, i_list, edge_list):
		mat = coo_matrix((edge_list, (u_list, i_list)), shape=(self.config.data.user_num, self.config.data.item_num), dtype=np.float32)
		sparse_ui_tensor = DataHandler.makeTorchAdj(mat, self.config.data.user_num, self.config.data.item_num, self.device)
		return sparse_ui_tensor

	def trainEpoch(self):
		self.handler.trainData.negSampling()

		ep_loss, ep_rec_loss, ep_reg_loss, ep_cl_loss = 0, 0, 0, 0
		image_diff_loss, text_diff_loss, audio_diff_loss = 0, 0, 0
		train_steps = len(self.handler.trainData) // self.config.train.batch
		diffusion_steps = len(self.handler.diffusionData) // self.config.train.batch

		main_log.info('Diffusion model training')
		for i, batch_data in enumerate(self.handler.diffusionLoader):
			# batch: list(tensor), batch[0]: (batch_size, item_num), batch[1]: (batch_size, )
			batch_u_items, batch_u_idxs = batch_data

			i_embs = self.model.getItemEmbs().detach()
			image_feats = self.model.getImageFeats().detach()
			text_feats = self.model.getTextFeats().detach()

			image_fit_noise_loss, image_refact_ui_loss = self.diffusion_model.training_losses(self.image_denoise_model, batch_u_items, i_embs, image_feats)
			text_fit_noise_loss, text_refact_ui_loss = self.diffusion_model.training_losses(self.text_denoise_model, batch_u_items, i_embs, text_feats)

			loss_image = image_fit_noise_loss.mean() + image_refact_ui_loss.mean() * self.config.hyper.e_loss
			loss_text = text_fit_noise_loss.mean() + text_refact_ui_loss.mean() * self.config.hyper.e_loss

			image_diff_loss += loss_image.item()
			text_diff_loss += loss_text.item()

			# optimizer
			self.image_denoise_opt.zero_grad()
			self.text_denoise_opt.zero_grad()

			if self.config.data.name == 'tiktok':
				audio_feats = self.model.getAudioFeats()
				assert audio_feats is not None
				audio_feats = audio_feats.detach()
				self.audio_denoise_opt.zero_grad()
				audio_fit_noist_loss, audio_refact_ui_loss = self.diffusion_model.training_losses(self.audio_denoise_model, batch_u_items, i_embs, audio_feats)
				loss_audio = audio_fit_noist_loss.mean() + audio_refact_ui_loss.mean() * self.config.hyper.e_loss
				audio_diff_loss += loss_audio.item()
				batch_diff_loss = loss_image + loss_text + loss_audio
			else:
				batch_diff_loss = loss_image + loss_text

			batch_diff_loss.backward()

			self.image_denoise_opt.step()
			self.text_denoise_opt.step()
			if self.config.data.name == 'tiktok':
				self.audio_denoise_opt.step()

		main_log.info('Re-build multimodal UI matrix')
		with torch.no_grad():
			u_list_image = []
			i_list_image = []
			edge_list_image = []

			u_list_text = []
			i_list_text = []
			edge_list_text = []

			u_list_audio = []
			i_list_audio = []
			edge_list_audio = []

			for batch_data in self.handler.diffusionLoader:
				batch_u_items: Tensor = batch_data[0]
				batch_u_idxs: Tensor = batch_data[1]
				topk = self.config.hyper.rebuild_k

				# image (batch, item)
				denoised_batch = self.diffusion_model.backward_steps(self.image_denoise_model, batch_u_items, self.config.hyper.sampling_steps, self.config.train.sampling_noise)
				_u_topk_items, indices = torch.topk(denoised_batch, k=topk)  # (batch, k)

				for i in range(batch_u_idxs.shape[0]):
					for j in range(indices[i].shape[0]):
						u_list_image.append(int(batch_u_idxs[i].cpu().numpy()))
						i_list_image.append(int(indices[i][j].cpu().numpy()))
						edge_list_image.append(1.0)

				# text
				denoised_batch = self.diffusion_model.backward_steps(self.text_denoise_model, batch_u_items, self.config.hyper.sampling_steps, self.config.train.sampling_noise)
				_u_topk_items, indices = torch.topk(denoised_batch, k=self.config.hyper.rebuild_k)

				for i in range(batch_u_idxs.shape[0]):
					for j in range(indices[i].shape[0]): 
						u_list_text.append(int(batch_u_idxs[i].cpu().numpy()))
						i_list_text.append(int(indices[i][j].cpu().numpy()))
						edge_list_text.append(1.0)

				if self.config.data.name == 'tiktok':
					# audio
					denoised_batch = self.diffusion_model.backward_steps(self.audio_denoise_model, batch_u_items, self.config.hyper.sampling_steps, self.config.train.sampling_noise)
					_u_topk_items, indices = torch.topk(denoised_batch, k=self.config.hyper.rebuild_k)

					for i in range(batch_u_idxs.shape[0]):
						for j in range(indices[i].shape[0]): 
							u_list_audio.append(int(batch_u_idxs[i].cpu().numpy()))
							i_list_audio.append(int(indices[i][j].cpu().numpy()))
							edge_list_audio.append(1.0)

			# image
			u_list_image = np.array(u_list_image)
			i_list_image = np.array(i_list_image)
			edge_list_image = np.array(edge_list_image)
			self.image_adj = self.makeTorchAdj(u_list_image, i_list_image, edge_list_image)
			self.image_adj = self.model.edgeDropper(self.image_adj)

			# text
			u_list_text = np.array(u_list_text)
			i_list_text = np.array(i_list_text)
			edge_list_text = np.array(edge_list_text)
			self.text_adj = self.makeTorchAdj(u_list_text, i_list_text, edge_list_text)
			self.text_adj = self.model.edgeDropper(self.text_adj)

			if self.config.data.name == 'tiktok':
				# audio
				u_list_audio = np.array(u_list_audio)
				i_list_audio = np.array(i_list_audio)
				edge_list_audio = np.array(edge_list_audio)
				self.audio_adj = self.makeTorchAdj(u_list_audio, i_list_audio, edge_list_audio)
				self.audio_adj = self.model.edgeDropper(self.audio_adj)


		main_log.info('Joint training 🤝')
		for i, batch_data in enumerate(self.handler.trainLoader):
			users, pos_items, neg_items = batch_data
			users: Tensor = users.long().cuda(self.device)
			pos_items: Tensor = pos_items.long().cuda(self.device)
			neg_items: Tensor = neg_items.long().cuda(self.device)

			self.opt.zero_grad()

			if self.config.data.name == 'tiktok':
				final_user_embs, final_item_embs = self.model.gcn_MM(self.handler.torchBiAdj, self.image_adj, self.text_adj, self.audio_adj)
			else:
				final_user_embs, final_item_embs = self.model.gcn_MM(self.handler.torchBiAdj, self.image_adj, self.text_adj)
			
			u_embs = final_user_embs[users]
			pos_embs = final_item_embs[pos_items]
			neg_embs = final_item_embs[neg_items]

			scoreDiff = pairPredict(u_embs, pos_embs, neg_embs)
			bpr_loss = - (scoreDiff).sigmoid().log().sum() / self.config.train.batch
			reg_loss = l2_reg_loss(self.config.train.reg, [self.model.u_embs, self.model.i_embs], self.device)
			# reg_loss = self.model.reg_loss() * self.config.train.reg
			ep_rec_loss += bpr_loss.item()
			ep_reg_loss += reg_loss.item()

			#* Cross layer CL
			joint_embs = torch.cat([self.model.u_embs, self.model.i_embs], dim=0)
			all_embs = []
			all_embs_cl = joint_embs
			for k in range(3): # GCN Layers = 3
				joint_embs = torch.sparse.mm(self.handler.torchBiAdj, joint_embs)
				random_noise = torch.rand_like(joint_embs)
				joint_embs += torch.sign(joint_embs) * F.normalize(random_noise) # noise scale
				all_embs.append(joint_embs)
				if k == 0: # which layer to CL
					all_embs_cl = joint_embs
			final_embs = torch.mean(torch.stack(all_embs), dim=0)
			
			cl1_user_embs = final_embs[:self.config.data.user_num]
			cl1_item_embs = final_embs[self.config.data.user_num:]
			cl2_user_embs = all_embs_cl[:self.config.data.user_num]
			cl2_item_embs = all_embs_cl[self.config.data.user_num:]

			#* Cross CL Loss
			cross_cl_loss = (InfoNCE(cl1_user_embs, cl2_user_embs, users, self.config.hyper.temp) + InfoNCE(cl1_item_embs, cl2_item_embs, pos_items, self.config.hyper.temp)) * self.config.hyper.ssl_reg


			#* Modality view as the anchor
			if self.config.data.name == 'tiktok':
				result = self.model.gcn_MM_CL(self.handler.torchBiAdj, self.image_adj, self.text_adj, self.audio_adj)
				assert len(result) == 6
				u_image_embs, i_image_embs, u_text_embs, i_text_embs, u_audio_embs, i_audio_embs = result
				# pairwise CL: image-text, image-audio, text-audio
				cross_modal_cl_loss = (InfoNCE(u_image_embs, u_text_embs, users, self.config.hyper.temp) + InfoNCE(i_image_embs, i_text_embs, pos_items, self.config.hyper.temp)) * self.config.hyper.ssl_reg
				cross_modal_cl_loss += (InfoNCE(u_image_embs, u_audio_embs, users, self.config.hyper.temp) + InfoNCE(i_image_embs, i_audio_embs, pos_items, self.config.hyper.temp)) * self.config.hyper.ssl_reg
				cross_modal_cl_loss += (InfoNCE(u_text_embs, u_audio_embs, users, self.config.hyper.temp) + InfoNCE(i_text_embs, i_audio_embs, pos_items, self.config.hyper.temp)) * self.config.hyper.ssl_reg
				# cross_modal_cl_loss = (contrastLoss(u_image_embs, u_text_embs, users, self.config.hyper.temp) + contrastLoss(i_image_embs, i_text_embs, pos_items, self.config.hyper.temp)) * self.config.hyper.ssl_reg
				# cross_modal_cl_loss += (contrastLoss(u_image_embs, u_audio_embs, users, self.config.hyper.temp) + contrastLoss(i_image_embs, i_audio_embs, pos_items, self.config.hyper.temp)) * self.config.hyper.ssl_reg
				# cross_modal_cl_loss += (contrastLoss(u_text_embs, u_audio_embs, users, self.config.hyper.temp) + contrastLoss(i_text_embs, i_audio_embs, pos_items, self.config.hyper.temp)) * self.config.hyper.ssl_reg
			else:
				result = self.model.gcn_MM_CL(self.handler.torchBiAdj, self.image_adj, self.text_adj)
				assert len(result) == 4
				u_image_embs, i_image_embs, u_text_embs, i_text_embs = result
				# only one CL: image-text
				cross_modal_cl_loss = (InfoNCE(u_image_embs, u_text_embs, users, self.config.hyper.temp) + InfoNCE(i_image_embs, i_text_embs, pos_items, self.config.hyper.temp)) * self.config.hyper.ssl_reg
				# cross_modal_cl_loss = (contrastLoss(u_image_embs, u_text_embs, users, self.config.hyper.temp) + contrastLoss(i_image_embs, i_text_embs, pos_items, self.config.hyper.temp)) * self.config.hyper.ssl_reg

			#* Main view as the anchor
			# main_cl_loss = (contrastLoss(final_user_embs, u_image_embs, users, self.config.hyper.temp) + contrastLoss(final_item_embs, i_image_embs, pos_items, self.config.hyper.temp)) * self.config.hyper.ssl_reg
			# main_cl_loss += (contrastLoss(final_user_embs, u_text_embs, users, self.config.hyper.temp) + contrastLoss(final_item_embs, i_text_embs, pos_items, self.config.hyper.temp)) * self.config.hyper.ssl_reg
			# if self.config.data.name == 'tiktok':
			# 	main_cl_loss += (contrastLoss(final_user_embs, u_audio_embs, users, self.config.hyper.temp) + contrastLoss(final_item_embs, i_audio_embs, pos_items, self.config.hyper.temp)) * self.config.hyper.ssl_reg # type: ignore
			main_cl_loss = (InfoNCE(final_user_embs, u_image_embs, users, self.config.hyper.temp) + InfoNCE(final_item_embs, i_image_embs, pos_items, self.config.hyper.temp)) * self.config.hyper.ssl_reg
			main_cl_loss += (InfoNCE(final_user_embs, u_text_embs, users, self.config.hyper.temp) + InfoNCE(final_item_embs, i_text_embs, pos_items, self.config.hyper.temp)) * self.config.hyper.ssl_reg
			if self.config.data.name == 'tiktok':
				main_cl_loss += (InfoNCE(final_user_embs, u_audio_embs, users, self.config.hyper.temp) + InfoNCE(final_item_embs, i_audio_embs, pos_items, self.config.hyper.temp)) * self.config.hyper.ssl_reg # type: ignore

			# if self.config.base.cl_method == 1: #! Only one of the two CL methods was used.
			# 	cl_loss = main_cl_loss
			# else:
			# 	cl_loss = cross_modal_cl_loss
			
			#* 同时启用两种对比学习
			cl_loss = cross_modal_cl_loss + main_cl_loss + cross_cl_loss
			# --------------------
			ep_cl_loss += cl_loss.item()

			batch_joint_loss =  bpr_loss + reg_loss + cl_loss
			ep_loss += batch_joint_loss.item()

			batch_joint_loss.backward()
			self.opt.step()

			# if i == train_steps:
			# 	print(f"bpr: {bpr_loss.item():.4f}, reg: {reg_loss.item():.4f}, cl: {cl_loss.item():.4f}")

		result = dict()
		result['Loss'] = ep_loss / train_steps
		result['BPR Loss'] = ep_rec_loss / train_steps
		result['reg loss'] = ep_reg_loss / train_steps
		result['CL loss'] = ep_cl_loss / train_steps
		result['Di image loss'] = image_diff_loss / diffusion_steps
		result['Di text loss'] = text_diff_loss / diffusion_steps
		if self.config.data.name == 'tiktok':
			result['Di audio loss'] = audio_diff_loss / diffusion_steps
		return result

	def testEpoch(self):
		testData = self.handler.testData
		testLoader = self.handler.testLoader
		epRecall, epNdcg, epPrecision = [0] * 3
		i = 0
		data_length = len(testData)
		test_steps = len(testData) // self.config.train.test_batch

		if self.config.data.name == 'tiktok':
			user_embs, item_embs = self.model.gcn_MM(self.handler.torchBiAdj, self.image_adj, self.text_adj, self.audio_adj)
		else:
			user_embs, item_embs = self.model.gcn_MM(self.handler.torchBiAdj, self.image_adj, self.text_adj)

		for usr, trainMask in testLoader:
			i += 1
			usr: Tensor = usr.long().cuda(self.device)
			trainMask: Tensor = trainMask.cuda(self.device)
			# batch users' scores for all items -> (batch, dim) @ (dim, item) = (batch, item)
			# 1-trainMask: reverse train mat (block train set) -> (batch_user, item)
			# trainMase*1e-8: set the score of train items to minimum to ensure the top-k items will not be selected
			predict = torch.mm(user_embs[usr], torch.transpose(item_embs, 1, 0)) * (1 - trainMask) - trainMask * 1e8
			_, top_idxs = torch.topk(predict, self.config.base.topk)  # (batch, topk)
			recall, ndcg, precision = self.calcRes(top_idxs.cpu().numpy(), testData.test_user_its, usr)
			epRecall += recall
			epNdcg += ndcg
			epPrecision += precision
			# olog(f"Step {i}/{test_steps}: recall = {recall:.2f}, ndcg = {ndcg:.2f} , precision = {precision:.2f}")
		ret = dict()
		ret['Recall'] = epRecall / data_length
		ret['NDCG'] = epNdcg / data_length
		ret['Precision'] = epPrecision / data_length
		return ret

	def calcRes(self, top_idxs: np.ndarray, test_u_its: list, users: Tensor):
		"""
		
		Args:
			top_idxs (np.ndarray): top-k items' index of test batch users: (test_batch, topk)
			test_u_its (list): test users' interactions: (test_batch, item)
			users (Tensor): test batch users' index: (test_batch, )
		"""
		assert top_idxs.shape[0] == len(users)
		allRecall = allNdcg = allPrecision = 0
		for i in range(len(users)):
			u_rec_list = list(top_idxs[i]) # =topk
			u_its = test_u_its[users[i]]
			tstNum = len(u_its)
			maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, self.config.base.topk))])
			recall_hits = dcg = precision_hits = 0
			for item in u_its:
				if item in u_rec_list:
					recall_hits += 1
					dcg += np.reciprocal(np.log2(u_rec_list.index(item) + 2))
					precision_hits += 1
			recall = recall_hits / tstNum
			ndcg = dcg / maxDcg
			precision = precision_hits / self.config.base.topk
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
	parser = argparse.ArgumentParser(description='Model Configs')
	parser.add_argument('--config', '-c', default='conf/test.toml', type=str, help='config file path')
	args = parser.parse_args()
	try:
		config = load_config(args.config)
		print("Load configuration file successfully👌")
	except Exception as e:
		print(f"Error loading configuration file: {e}")
		exit(1)

	seed_it(config.base.seed)

	main_log = Log('main', config.data.name)
	main_log.info('Start')
	data_handler = DataHandler(config)

	main_log.info('Load Data')
	data_handler.LoadData()
	
	coach = Coach(data_handler, config)
	coach.run()