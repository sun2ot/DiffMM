import torch
from torch import Tensor
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from Utils.Log import Log
from Conf import load_config, Config
from Model import Model, GaussianDiffusion, Denoise
from DataHandler import DataHandler
import numpy as np
from Utils.Utils import *
import os
import random
from scipy.sparse import coo_matrix
import ast
import argparse
from sklearn.metrics.pairwise import cosine_similarity

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
			result_str += f"{metric}={val:.5f}, "
		result_str = result_str[:-2] + '  ' #? del blank and `:`?
		return result_str
	
	def save_max(self, new: list, old: list) -> list:
		"""Update the maximum indicator. More than half of updates are marked as new best"""
		re = []
		for i,j in zip(new, old):
			if i > j:
				re.append(i)
			else:
				re.append(j)
		return re

	def run(self):
		self.prepareModel()
		main_log.info('Model Initialized ✅')

		recallMax, ndcgMax, precisionMax = 0, 0, 0
		his_max = [0, 0, 0]
		bestEpoch = 0  #todo: early stop

		main_log.info('Start training 🚀')
		try:
			for epoch in range(0, self.config.train.epoch):
				tstFlag = (epoch % self.config.train.tstEpoch == 0)
				result = self.trainEpoch()

				if self.config.train.use_lr_scheduler:
					self.model_scheduler.step()
					# ----------- Ablation3: KNN -----------
					self.image_scheduler.step()
					self.text_scheduler.step()
					if self.config.data.name == 'tiktok':
						self.audio_scheduler.step()
					# ----------- Ablation3: KNN -----------
				
				main_log.info(self.makePrint('⏩ Train', epoch, result))
				if tstFlag:
					result = self.testEpoch()
					his_max = self.save_max([result['Recall'], result['NDCG'], result['Precision']], his_max)
					if result['Recall'] > recallMax:
						recallMax = result['Recall']
						ndcgMax = result['NDCG']
						precisionMax = result['Precision']
						bestEpoch = epoch
					main_log.info(self.makePrint('🧪 Test', epoch, result))
				main_log.info(f"💡 Current best: Epoch: {bestEpoch}, Recall: {recallMax:.5f}({his_max[0]:.5f}), NDCG: {ndcgMax:.5f}({his_max[1]:.5f}), Precision: {precisionMax:.5f}({his_max[2]:.5f})")
			main_log.info(f"Best epoch: {bestEpoch}, Recall: {recallMax:.5f}({his_max[0]:.5f}), NDCG: {ndcgMax:.5f}({his_max[1]:.5f}), Precision: {precisionMax:.5f}({his_max[2]:.5f})")
		except KeyboardInterrupt:
			main_log.info('🈲 Training interrupted by user!')
			main_log.info(f"💡 Current best: Epoch: {bestEpoch}, Recall: {recallMax:.5f}({his_max[0]:.5f}), NDCG: {ndcgMax:.5f}({his_max[1]:.5f}), Precision: {precisionMax:.5f}({his_max[2]:.5f})")


	def prepareModel(self):
		"""Init DiffMM, Diffusion, Denoise Models"""
		if self.config.data.name == 'tiktok':
			self.model = Model(self.config, self.handler.image_feats.detach(), self.handler.text_feats.detach(), self.handler.audio_feats.detach()).cuda(self.device)
		else:
			self.model = Model(self.config, self.handler.image_feats.detach(), self.handler.text_feats.detach()).cuda(self.device)

		self.opt = Adam(self.model.parameters(), lr=self.config.train.lr, weight_decay=0)
		self.model_scheduler = CosineAnnealingLR(self.opt, T_max=self.config.train.epoch, eta_min=1e-4)

		self.diffusion_model = GaussianDiffusion(self.config).cuda(self.device)
		
		out_dims = ast.literal_eval(self.config.base.denoise_dim) + [self.config.data.item_num] # [denoise_dim, item_num]
		in_dims = out_dims[::-1]  # [item_num, denoise_dim]
		self.image_denoise_model = Denoise(in_dims, out_dims, config).cuda(self.device)
		self.image_denoise_opt = Adam(self.image_denoise_model.parameters(), lr=self.config.train.lr, weight_decay=0)
		self.image_scheduler = CosineAnnealingLR(self.image_denoise_opt, T_max=self.config.train.epoch, eta_min=1e-4)

		self.text_denoise_model = Denoise(in_dims, out_dims, config).cuda(self.device)
		self.text_denoise_opt = Adam(self.text_denoise_model.parameters(), lr=self.config.train.lr, weight_decay=0)
		self.text_scheduler = CosineAnnealingLR(self.text_denoise_opt, T_max=self.config.train.epoch, eta_min=1e-4)

		if self.config.data.name == 'tiktok':
			self.audio_denoise_model = Denoise(in_dims, out_dims, config).cuda(self.device)
			self.audio_denoise_opt = Adam(self.audio_denoise_model.parameters(), lr=self.config.train.lr, weight_decay=0)
			self.audio_scheduler = CosineAnnealingLR(self.audio_denoise_opt, T_max=self.config.train.epoch, eta_min=1e-4)


	def makeTorchAdj(self, u_list, i_list, edge_list):
		mat = coo_matrix((edge_list, (u_list, i_list)), shape=(self.config.data.user_num, self.config.data.item_num), dtype=np.float32)
		sparse_ui_tensor = DataHandler.makeTorchAdj(mat, self.config.data.user_num, self.config.data.item_num, self.device)
		return sparse_ui_tensor
	
	def build_knn_adj(self, user_pos_items, item_feats, topk_per_user):
		"""Ablation3: KNN to generate modality user-item adjacency matrix"""
		user_proto = np.array([
			item_feats[items].mean(axis=0) if len(items) > 0 else np.zeros(item_feats.shape[1])
			for items in user_pos_items
		])  # shape=(user_num, feat_dim)

		sim = cosine_similarity(user_proto, item_feats)  # (user_num, item_num)

		u_list, i_list, vals = [], [], []
		for u in range(sim.shape[0]):
			# 取topk有序索引
			idx = np.argsort(-sim[u])[:topk_per_user]
			u_list.extend([u] * topk_per_user)
			i_list.extend(idx.tolist())
			vals.extend([1.0] * topk_per_user)
		return np.array(u_list), np.array(i_list), np.array(vals)

	def trainEpoch(self):
		self.handler.trainData.negSampling()

		ep_loss, ep_rec_loss, ep_reg_loss, ep_cl_loss = 0, 0, 0, 0
		image_diff_loss, text_diff_loss, audio_diff_loss = 0, 0, 0
		train_steps = len(self.handler.trainData) // self.config.train.batch
		diffusion_steps = len(self.handler.diffusionData) // self.config.train.batch

		main_log.info('Diffusion model training')
		for i, batch_data in enumerate(self.handler.diffusionLoader):
			# batch: list(tensor), batch[0]: (batch_size, item_num), batch[1]: (batch_size, )
			batch_u_items = batch_data[0]

			i_embs = self.model.getItemEmbs()
			image_feats = self.model.getImageFeats().detach()
			text_feats = self.model.getTextFeats().detach()

			batch_image_loss: Tensor = self.diffusion_model.training_losses(self.image_denoise_model, batch_u_items, i_embs, image_feats)
			loss_image = batch_image_loss.mean()
			image_diff_loss += loss_image.item()

			batch_text_loss: Tensor = self.diffusion_model.training_losses(self.text_denoise_model, batch_u_items, i_embs, text_feats)
			loss_text = batch_text_loss.mean()
			text_diff_loss += loss_text.item()

			# optimizer
			self.image_denoise_opt.zero_grad()
			self.text_denoise_opt.zero_grad()

			if self.config.data.name == 'tiktok':
				audio_feats = self.model.getAudioFeats()
				assert audio_feats is not None
				audio_feats = audio_feats.detach()
				self.audio_denoise_opt.zero_grad()
				batch_audio_loss: Tensor = self.diffusion_model.training_losses(self.audio_denoise_model, batch_u_items, i_embs, audio_feats)
				loss_audio = batch_audio_loss.mean()
				audio_diff_loss += loss_audio.item()

				# Normalize the losses before summing
				total_loss = loss_image.item() + loss_text.item() + loss_audio.item()
				batch_diff_loss = (loss_image + loss_text + loss_audio)/total_loss
				image_diff_loss /= total_loss
				text_diff_loss /= total_loss
				audio_diff_loss /= total_loss
			else:
				# Normalize the losses before summing
				total_loss = loss_image.item() + loss_text.item()
				batch_diff_loss = (loss_image + loss_text)/total_loss
				image_diff_loss /= total_loss
				text_diff_loss /= total_loss

			batch_diff_loss.backward()

			self.image_denoise_opt.step()
			self.text_denoise_opt.step()
			if self.config.data.name == 'tiktok':
				self.audio_denoise_opt.step()


		main_log.info('Re-build multimodal UI matrix')
		with torch.no_grad():
			# every modal's u_list/i_list/edge_list for creating adjacency matrix
			modality_names = ['image', 'text']
			if self.config.data.name == 'tiktok':
				modality_names.append('audio')
			u_list_dict = {m: [] for m in modality_names}
			i_list_dict = {m: [] for m in modality_names}
			edge_list_dict = {m: [] for m in modality_names}
			denoise_model_dict = {
				'image': self.image_denoise_model,
				'text': self.text_denoise_model,
			}
			if self.config.data.name == 'tiktok':
				denoise_model_dict['audio'] = self.audio_denoise_model

			for batch_data in self.handler.diffusionLoader:
				batch_u_items: Tensor = batch_data[0]
				batch_u_idxs: np.ndarray = batch_data[1].cpu().numpy()

				user_degrees = self.handler.getUserDegrees()
				topk_values = user_degrees[batch_u_idxs]

				for m in modality_names:
					denoised_batch = self.diffusion_model.generate_view(
						denoise_model_dict[m],
						batch_u_items,
						self.config.hyper.sampling_step
					)
					for i in range(batch_u_idxs.shape[0]):
						user_topk = topk_values[i]
						_, indices = torch.topk(denoised_batch[i], k=user_topk)  # (batch_size, topk)
						for j in range(indices.shape[0]):
							u_list_dict[m].append(batch_u_idxs[i])
							i_list_dict[m].append(int(indices[j]))
							edge_list_dict[m].append(1.0)

			# make torch sparse adjacency matrix
			self.image_adj = self.makeTorchAdj(
				np.array(u_list_dict['image']),
				np.array(i_list_dict['image']),
				np.array(edge_list_dict['image'])
			)
			# self.image_adj = self.model.edgeDropper(self.image_adj)

			self.text_adj = self.makeTorchAdj(
				np.array(u_list_dict['text']),
				np.array(i_list_dict['text']),
				np.array(edge_list_dict['text'])
			)
			# self.text_adj = self.model.edgeDropper(self.text_adj)

			if self.config.data.name == 'tiktok':
				self.audio_adj = self.makeTorchAdj(
					np.array(u_list_dict['audio']),
					np.array(i_list_dict['audio']),
					np.array(edge_list_dict['audio'])
				)
				# self.audio_adj = self.model.edgeDropper(self.audio_adj)

		# ------------------ Ablation3 ------------------
		# main_log.info('Re-build multimodal UI matrix (KNN)')
		# with torch.no_grad():
		# 	# image
		# 	u_i, i_i, v_i = self.build_knn_adj(
		# 		self.handler.trainData.user_pos_items,
		# 		self.handler.image_feats.detach().cpu().numpy(),
		# 		self.config.hyper.knn_topk
		# 	)
		# 	self.image_adj = self.makeTorchAdj(
		# 		np.array(u_i), np.array(i_i), np.array(v_i)
		# 	)

		# 	# text
		# 	u_t, i_t, v_t = self.build_knn_adj(
		# 		self.handler.trainData.user_pos_items,
		# 		self.handler.text_feats.detach().cpu().numpy(),
		# 		self.config.hyper.knn_topk
		# 	)
		# 	self.text_adj = self.makeTorchAdj(
		# 		np.array(u_t), np.array(i_t), np.array(v_t)
		# 	)
			
		# 	# audio
		# 	if self.config.data.name == 'tiktok':
		# 		u_a, i_a, v_a = self.build_knn_adj(
		# 			self.handler.trainData.user_pos_items,
		# 			self.handler.audio_feats.detach().cpu().numpy(),
		# 			self.config.hyper.knn_topk
		# 		)
		# 		self.audio_adj = self.makeTorchAdj(
		# 			np.array(u_a), np.array(i_a), np.array(v_a)
		# 		)
		# ------------------ Ablation3 ------------------


		main_log.info('Joint training 🤝')
		for i, batch_data in enumerate(self.handler.trainLoader):
			users, pos_items, neg_items = batch_data
			users: Tensor = users.long().cuda(self.device)
			pos_items: Tensor = pos_items.long().cuda(self.device)
			neg_items: Tensor = neg_items.long().cuda(self.device)

			if self.config.data.name == 'tiktok':
				gcn_output = self.model.gcn_MM(self.handler.torchBiAdj, self.image_adj, self.text_adj, self.audio_adj)
				final_user_embs, final_item_embs = gcn_output.u_final_embs, gcn_output.i_final_embs
			else:
				gcn_output = self.model.gcn_MM(self.handler.torchBiAdj, self.image_adj, self.text_adj)
				final_user_embs, final_item_embs = gcn_output.u_final_embs, gcn_output.i_final_embs
			
			u_embs = final_user_embs[users]
			pos_embs = final_item_embs[pos_items]
			neg_embs = final_item_embs[neg_items]

			rec_loss = bpr_loss(u_embs, pos_embs, neg_embs)
			reg_loss = l2_reg_loss(self.config.train.reg, [self.model.u_embs, self.model.i_embs], self.device)
			ep_rec_loss += rec_loss.item()
			ep_reg_loss += reg_loss.item()

			#* Cross layer CL
			joint_embs = torch.cat([self.model.u_embs, self.model.i_embs], dim=0)
			all_embs = []
			all_embs_cl = joint_embs
			for k in range(3): # GCN Layers = 3
				joint_embs = torch.sparse.mm(self.handler.torchBiAdj, joint_embs)
				random_noise = torch.rand_like(joint_embs)
				joint_embs += torch.sign(joint_embs) * F.normalize(random_noise) * self.config.hyper.noise_degree
				all_embs.append(joint_embs)
				if k == 0: # which layer to CL
					all_embs_cl = joint_embs
			final_embs = torch.mean(torch.stack(all_embs), dim=0)
			
			cl1_user_embs = final_embs[:self.config.data.user_num]
			cl1_item_embs = final_embs[self.config.data.user_num:]
			cl2_user_embs = all_embs_cl[:self.config.data.user_num]
			cl2_item_embs = all_embs_cl[self.config.data.user_num:]

			#* Cross CL Loss
			cross_cl_loss = (InfoNCE(cl1_user_embs, cl2_user_embs, users, self.config.hyper.cross_cl_temp) + InfoNCE(cl1_item_embs, cl2_item_embs, pos_items, self.config.hyper.cross_cl_temp)) * self.config.hyper.cross_cl_rate
			cl_loss = cross_cl_loss

			# Ablation1
			# cl_loss = 0

			# ----------- Ablation2 -----------
			if self.config.data.name == 'tiktok':
				u_image_embs, i_image_embs = gcn_output.u_image_embs, gcn_output.i_image_embs
				u_text_embs, i_text_embs = gcn_output.u_text_embs, gcn_output.i_text_embs
				u_audio_embs, i_audio_embs = gcn_output.u_audio_embs, gcn_output.i_audio_embs
				assert u_audio_embs is not None and i_audio_embs is not None
				if self.config.base.cl_method == 1:
					# pairwise CL: image-text, image-audio, text-audio
					cross_modal_cl_loss = (InfoNCE(u_image_embs, u_text_embs, users, self.config.hyper.modal_cl_temp) + InfoNCE(i_image_embs, i_text_embs, pos_items, self.config.hyper.modal_cl_temp)) * self.config.hyper.modal_cl_rate
					cross_modal_cl_loss += (InfoNCE(u_image_embs, u_audio_embs, users, self.config.hyper.modal_cl_temp) + InfoNCE(i_image_embs, i_audio_embs, pos_items, self.config.hyper.modal_cl_temp)) * self.config.hyper.modal_cl_rate
					cross_modal_cl_loss += (InfoNCE(u_text_embs, u_audio_embs, users, self.config.hyper.modal_cl_temp) + InfoNCE(i_text_embs, i_audio_embs, pos_items, self.config.hyper.modal_cl_temp)) * self.config.hyper.modal_cl_rate
					cl_loss += cross_modal_cl_loss
				else:
					# only one CL: image-text
					main_cl_loss = (InfoNCE(final_user_embs, u_image_embs, users, self.config.hyper.modal_cl_temp) + InfoNCE(final_item_embs, i_image_embs, pos_items, self.config.hyper.modal_cl_temp)) * self.config.hyper.modal_cl_rate
					main_cl_loss += (InfoNCE(final_user_embs, u_text_embs, users, self.config.hyper.modal_cl_temp) + InfoNCE(final_item_embs, i_text_embs, pos_items, self.config.hyper.modal_cl_temp)) * self.config.hyper.modal_cl_rate
					main_cl_loss += (InfoNCE(final_user_embs, u_audio_embs, users, self.config.hyper.modal_cl_temp) + InfoNCE(final_item_embs, i_audio_embs, pos_items, self.config.hyper.modal_cl_temp)) * self.config.hyper.modal_cl_rate
					cl_loss += main_cl_loss
			else:
				u_image_embs, i_image_embs = gcn_output.u_image_embs, gcn_output.i_image_embs
				u_text_embs, i_text_embs = gcn_output.u_text_embs, gcn_output.i_text_embs
				if self.config.base.cl_method == 1: #! Only one of the two CL methods was used.
					cross_modal_cl_loss = (InfoNCE(u_image_embs, u_text_embs, users, self.config.hyper.modal_cl_temp) + InfoNCE(i_image_embs, i_text_embs, pos_items, self.config.hyper.modal_cl_temp)) * self.config.hyper.modal_cl_rate
					cl_loss += cross_modal_cl_loss
				else:
					#* Main view as the anchor
					main_cl_loss = (InfoNCE(final_user_embs, u_image_embs, users, self.config.hyper.modal_cl_temp) + InfoNCE(final_item_embs, i_image_embs, pos_items, self.config.hyper.modal_cl_temp)) * self.config.hyper.modal_cl_rate
					main_cl_loss += (InfoNCE(final_user_embs, u_text_embs, users, self.config.hyper.modal_cl_temp) + InfoNCE(final_item_embs, i_text_embs, pos_items, self.config.hyper.modal_cl_temp)) * self.config.hyper.modal_cl_rate
					cl_loss += main_cl_loss
			# ----------- Ablation2 -----------

			ep_cl_loss += cl_loss.item()

			batch_joint_loss =  rec_loss + reg_loss + cl_loss
			ep_loss += batch_joint_loss.item()
			
			self.opt.zero_grad()
			batch_joint_loss.backward()
			self.opt.step()

		result = dict()
		result['Loss'] = ep_loss / train_steps
		result['BPR Loss'] = ep_rec_loss / train_steps
		result['reg loss'] = ep_reg_loss / train_steps
		result['CL loss'] = ep_cl_loss / train_steps
		result['image loss'] = image_diff_loss / diffusion_steps
		result['text loss'] = text_diff_loss / diffusion_steps
		if self.config.data.name == 'tiktok':
			result['audio loss'] = audio_diff_loss / diffusion_steps
		return result

	def testEpoch(self):
		testData = self.handler.testData
		testLoader = self.handler.testLoader
		epRecall, epNdcg, epPrecision = [0] * 3
		i = 0
		data_length = len(testData)

		if self.config.data.name == 'tiktok':
			gcn_output = self.model.gcn_MM(self.handler.torchBiAdj, self.image_adj, self.text_adj, self.audio_adj)
		else:
			gcn_output = self.model.gcn_MM(self.handler.torchBiAdj, self.image_adj, self.text_adj)
		user_embs, item_embs = gcn_output.u_final_embs, gcn_output.i_final_embs

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
	random.seed(seed)
	os.environ["PYTHONSEED"] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Model Configs')
	parser.add_argument('--config', '-c', default='conf/test.toml', type=str, help='config file path')
	args = parser.parse_args()
	try:
		config = load_config(args.config)
		print(f"Load configuration ({config.data.name}) file successfully👌")
	except Exception as e:
		print(f"Error loading configuration file: {e}")
		exit(1)

	seed_it(config.base.seed)

	main_log = Log('main', config.data.name)
	main_log.info('Start')
	main_log.info("Configuration Details:")
	for section, options in config.__dict__.items():
		if isinstance(options, dict):
			main_log.info(f"[{section}]")
			for key, value in options.items():
				main_log.info(f"  {key}: {value}")
		else:
			main_log.info(f"{section}: {options}")
	data_handler = DataHandler(config)

	main_log.info('Load Data')
	data_handler.LoadData()
	
	coach = Coach(data_handler, config)
	coach.run()

	# hyperparameters experiments

	# sampling_step = range(5)

	# for p in sampling_step:
	# 	config.hyper.sampling_step = p
	# 	main_log = Log('main', config.data.name)
	# 	main_log.info('Start')
	# 	main_log.info(f"================= sampling_step={p} ======================")
	# 	main_log.info("Configuration Details:")
	# 	for section, options in config.__dict__.items():
	# 		if isinstance(options, dict):
	# 			main_log.info(f"[{section}]")
	# 			for key, value in options.items():
	# 				main_log.info(f"  {key}: {value}")
	# 		else:
	# 			main_log.info(f"{section}: {options}")
	# 	data_handler = DataHandler(config)

	# 	main_log.info('Load Data')
	# 	data_handler.LoadData()
		
	# 	coach = Coach(data_handler, config)
	# 	coach.run()
	# 	main_log.info("\n\n=========================================\n\n")