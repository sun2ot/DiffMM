import torch
from torch import nn, Tensor
import torch.nn.functional as F
from Conf import Config
import numpy as np
import math
from Utils.Utils import *
from typing import Optional
from dataclasses import dataclass

init = nn.init.xavier_uniform_  # specific init func
uniformInit = nn.init.uniform   # func factory


class Model(nn.Module):
	def __init__(self, config: Config, image_embedding, text_embedding, audio_embedding=None):
		super(Model, self).__init__()

		self.config = config
		self.device = torch.device(f"cuda:{self.config.base.gpu}" if torch.cuda.is_available() else "cpu")
		self.u_embs = nn.Parameter(init(torch.empty(self.config.data.user_num, self.config.base.latdim)))
		self.i_embs = nn.Parameter(init(torch.empty(self.config.data.item_num, self.config.base.latdim)))
		self.layer = nn.Sequential(*[GCNLayer() for i in range(self.config.train.gnn_layer)])

		self.edgeDropper = SpAdjDropEdge(self.config.hyper.keepRate)

		if self.config.base.trans == 1:
			self.image_layer = nn.Linear(self.config.data.image_feat_dim, self.config.base.latdim)
			self.text_layer = nn.Linear(self.config.data.text_feat_dim, self.config.base.latdim)
		elif self.config.base.trans == 0:
			self.image_matrix = nn.Parameter(init(torch.empty(size=(self.config.data.image_feat_dim, self.config.base.latdim))))
			self.text_matrix = nn.Parameter(init(torch.empty(size=(self.config.data.text_feat_dim, self.config.base.latdim))))
		else:  # self.config.base.trans == 2
			self.image_matrix = nn.Parameter(init(torch.empty(size=(self.config.data.image_feat_dim, self.config.base.latdim))))
			self.text_layer = nn.Linear(self.config.data.text_feat_dim, self.config.base.latdim)
		if audio_embedding != None:
			if self.config.base.trans == 1:
				self.audio_layer = nn.Linear(self.config.data.audio_feat_dim, self.config.base.latdim)
			else:
				self.audio_matrix = nn.Parameter(init(torch.empty(size=(self.config.data.audio_feat_dim, self.config.base.latdim))))

		self.image_embedding = image_embedding
		self.text_embedding = text_embedding
		if audio_embedding != None:
			self.audio_embedding = audio_embedding
		else:
			self.audio_embedding = None
		
		# average weight
		if audio_embedding != None:
			self.modal_weight = nn.Parameter(torch.tensor([0.3333, 0.3333, 0.3333]))
		else:
			self.modal_weight = nn.Parameter(torch.tensor([0.5, 0.5]))
		self.softmax = nn.Softmax(dim=0)
		self.dropout = nn.Dropout(p=0.1)
		self.leakyrelu = nn.LeakyReLU(0.2)
				
	def getItemEmbs(self):
		return self.i_embs
	
	def getUserEmbs(self):
		return self.u_embs
	
	def getImageFeats(self) -> torch.Tensor:
		if self.config.base.trans == 0 or self.config.base.trans == 2:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_matrix))
			return image_feats
		else:
			return self.image_layer(self.image_embedding)
	
	def getTextFeats(self) -> torch.Tensor:
		if self.config.base.trans == 0:
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_matrix))
			return text_feats
		else:
			return self.text_layer(self.text_embedding)

	def getAudioFeats(self) -> None | torch.Tensor:
		if self.audio_embedding == None:
			return None
		else:
			if self.config.base.trans == 0:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_matrix))
			else:
				audio_feats = self.audio_layer(self.audio_embedding)
		return audio_feats

	def gcn_MM(self, adj: Tensor, image_adj: Tensor, text_adj: Tensor, audio_adj: Optional[Tensor] = None):
		"""
		Multimodal graph aggregation.

		Args:
			adj (Tensor): CF matrix
			image_adj (Tensor): CF matrix for image modal
			text_adj (Tensor): CF matrix for text modal
			audio_adj (Tensor): CF matrix for audio modal
		Returns:
			Tuple (Tensor, Tensor): (final_user_embs, final_item_embs)
		"""

		# @dataclass
		# class GCNOut:


		# Trans multimodal feats to 64 (latdim)
		if self.config.base.trans == 0:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_matrix))
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_matrix))
		elif self.config.base.trans == 1:
			image_feats = self.image_layer(self.image_embedding)
			text_feats = self.text_layer(self.text_embedding)
		else:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_matrix))
			text_feats = self.text_layer(self.text_embedding)

		if audio_adj != None:
			if self.config.base.trans == 0:
				assert self.audio_embedding != None
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_matrix))
			else:
				audio_feats = self.audio_layer(self.audio_embedding)
		else:
			audio_feats = None

		weight: nn.Parameter = self.softmax(self.modal_weight)

		# image
		# Eq.20 the third part
		image_adj_embs = torch.concat([self.u_embs, self.i_embs])  # (node, dim)
		image_adj_embs = torch.sparse.mm(image_adj, image_adj_embs)  # (node, dim)

		# Eq.20 the first part
		image_aware_embs = torch.concat([self.u_embs, F.normalize(image_feats)])  # (node, dim)
		image_aware_embs = torch.sparse.mm(adj, image_aware_embs)  # (node, dim)

		# Eq.20 the second part
		horder_image_aware_embs = torch.concat([image_aware_embs[:self.config.data.user_num], self.i_embs])
		horder_image_aware_embs = torch.sparse.mm(adj, horder_image_aware_embs)
		image_aware_embs += horder_image_aware_embs
		
		# text
		text_adj_embs = torch.concat([self.u_embs, self.i_embs])
		text_adj_embs = torch.sparse.mm(text_adj, text_adj_embs)

		text_aware_embs = torch.concat([self.u_embs, F.normalize(text_feats)])
		text_aware_embs = torch.sparse.mm(adj, text_aware_embs)

		horder_text_aware_embs = torch.concat([text_aware_embs[:self.config.data.user_num], self.i_embs])
		horder_text_aware_embs = torch.sparse.mm(adj, horder_text_aware_embs)
		text_aware_embs += horder_text_aware_embs

		# audio
		if audio_adj != None:
			audio_adj_embs = torch.concat([self.u_embs, self.i_embs])
			audio_adj_embs = torch.sparse.mm(audio_adj, audio_adj_embs)

			assert audio_feats != None
			audio_aware_embs = torch.concat([self.u_embs, F.normalize(audio_feats)])
			audio_aware_embs = torch.sparse.mm(adj, audio_aware_embs)

			horder_audio_aware_embs = torch.concat([audio_aware_embs[:self.config.data.user_num], self.i_embs])
			horder_audio_aware_embs = torch.sparse.mm(adj, horder_audio_aware_embs)
			audio_aware_embs += horder_audio_aware_embs
		else:
			audio_adj_embs, audio_aware_embs = None, None

		image_aware_embs += self.config.hyper.modal_adj_weight * image_adj_embs
		text_aware_embs += self.config.hyper.modal_adj_weight * text_adj_embs
		if audio_adj != None:
			assert audio_adj_embs != None
			audio_aware_embs += self.config.hyper.modal_adj_weight * audio_adj_embs
		
		if audio_adj == None:
			modal_embs = weight[0] * image_aware_embs + weight[1] * text_aware_embs
		else:
			modal_embs = weight[0] * image_aware_embs + weight[1] * text_aware_embs + weight[2] * audio_aware_embs

		final_embs = modal_embs
		embs_list = [final_embs]
		for gcn in self.layer:  # Eq.22
			final_embs = gcn(adj, embs_list[-1])
			embs_list.append(final_embs)
		final_embs = torch.stack(embs_list).sum(dim=0)

		final_embs = final_embs + self.config.hyper.residual_weight * F.normalize(modal_embs)

		return final_embs[:self.config.data.user_num], final_embs[self.config.data.user_num:]

	def gcn_MM_CL(self, adj: Tensor, image_adj: Tensor, text_adj: Tensor, audio_adj: Optional[Tensor] = None):
		"""
		Multimodal graph aggregation for CL task?

		Returns:
			image_u_embs, image_i_embs, text_u_embs, text_i_embs, Optional(audio_u_embs, audio_i_embs)
		"""
		if self.config.base.trans == 0:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_matrix))
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_matrix))
		elif self.config.base.trans == 1:
			image_feats = self.image_layer(self.image_embedding)
			text_feats = self.text_layer(self.text_embedding)
		else:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_matrix))
			text_feats = self.text_layer(self.text_embedding)

		if audio_adj != None:
			if self.config.base.trans == 0:
				assert self.audio_embedding != None
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_matrix))
			else:
				audio_feats = self.audio_layer(self.audio_embedding)
		else:
			audio_feats = None

		image_aware_embs = torch.concat([self.u_embs, F.normalize(image_feats)])
		image_aware_embs = torch.sparse.mm(image_adj, image_aware_embs)

		text_aware_embs = torch.concat([self.u_embs, F.normalize(text_feats)])
		text_aware_embs = torch.sparse.mm(text_adj, text_aware_embs)

		if audio_adj != None:
			assert audio_feats != None
			audio_aware_embs = torch.concat([self.u_embs, F.normalize(audio_feats)])
			audio_aware_embs = torch.sparse.mm(audio_adj, audio_aware_embs)
		else:
			audio_aware_embs = None

		image_modal_embs = image_aware_embs
		image_embs_list = [image_modal_embs]
		for gcn in self.layer:
			image_modal_embs = gcn(adj, image_embs_list[-1])
			image_embs_list.append(image_modal_embs)
		image_modal_embs = torch.sum(torch.stack(image_embs_list), dim=0)

		text_modal_embs = text_aware_embs
		text_embs_list = [text_modal_embs]
		for gcn in self.layer:
			text_modal_embs = gcn(adj, text_embs_list[-1])
			text_embs_list.append(text_modal_embs)
		text_modal_embs = torch.sum(torch.stack(text_embs_list), dim=0)

		if audio_adj != None:
			assert audio_aware_embs != None
			audio_modal_embs = audio_aware_embs
			audio_embs_list = [audio_modal_embs]
			for gcn in self.layer:
				audio_modal_embs = gcn(adj, audio_embs_list[-1])
				audio_embs_list.append(audio_modal_embs)
			audio_modal_embs = torch.sum(torch.stack(audio_embs_list), dim=0)
		else:
			audio_modal_embs = None

		if audio_adj == None:
			return image_modal_embs[:self.config.data.user_num], image_modal_embs[self.config.data.user_num:], text_modal_embs[:self.config.data.user_num], text_modal_embs[self.config.data.user_num:]
		else:
			assert audio_modal_embs != None
			return image_modal_embs[:self.config.data.user_num], image_modal_embs[self.config.data.user_num:], text_modal_embs[:self.config.data.user_num], text_modal_embs[self.config.data.user_num:], audio_modal_embs[:self.config.data.user_num], audio_modal_embs[self.config.data.user_num:]

	def reg_loss(self):
		"""calculate user and item embedding L2 reg loss"""
		loss = 0
		loss += self.u_embs.norm(2).square()
		loss += self.i_embs.norm(2).square()
		return loss

class GCNLayer(nn.Module):
	"""adi @ embs"""
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj: Tensor, embs: Tensor):
		"""sparse matrix multiplication"""
		return torch.sparse.mm(adj, embs)

class SpAdjDropEdge(nn.Module):
	def __init__(self, keepRate):
		super(SpAdjDropEdge, self).__init__()
		self.keepRate = keepRate

	def forward(self, adj):
		vals = adj._values()
		idxs = adj._indices()
		edgeNum = vals.size()
		mask = ((torch.rand(edgeNum) + self.keepRate).floor()).type(torch.bool)

		newVals = vals[mask] / self.keepRate
		newIdxs = idxs[:, mask]

		return torch.sparse_coo_tensor(newIdxs, newVals, adj.shape)
		
class Denoise(nn.Module):
	def __init__(self, in_dims: list[int], out_dims: list[int], config: Config, dropout=0.5):
		"""
		Denoiser class for the diffusion model.

		Args:
			in_dims (list): (item_num, 1000).
			out_dims (list): (1000, item_num)
			emb_size (int): 
			norm (bool): Whether to normalize the input
			dropout (float): 
		"""
		super(Denoise, self).__init__()
		self.device = torch.device(f"cuda:{config.base.gpu}" if torch.cuda.is_available() else "cpu")
		self.in_dims = in_dims
		self.out_dims = out_dims
		self.time_emb_dim = config.base.d_emb_size #* time embedding size for diffusion step
		self.norm = config.train.norm

		self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim) # (10, 10)

		in_dims_temp: list = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]  # (item_num+10, 1000)
		out_dims_temp: list = self.out_dims  # (1000, item_num)

		#! What is the dims for?
		# (item_num+10, 1000)
		self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
		# (1000, item_num)
		self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

		self.drop = nn.Dropout(dropout)
		self.init_weights()

	def init_weights(self):
		def initialize_layer(layer):
			"""Helper function to initialize weights and biases of a layer."""
			nn.init.xavier_normal_(layer.weight)
			if layer.bias is not None:  # Check if the layer has bias
				nn.init.normal_(layer.bias, mean=0.0, std=0.001)
		
		for layer in self.in_layers:
			initialize_layer(layer)
		for layer in self.out_layers:
			initialize_layer(layer)
		initialize_layer(self.emb_layer)

	def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor, mess_dropout=True) -> torch.Tensor:
		"""
		Denoise Layer

		Args:
			x (torch.Tensor): batch_u_items, (batch_size, item_num)
			timesteps (torch.Tensor): (batch_size)
		
		Returns:
			h (torch.Tensor): denoised view (batch_size, item_num)
		"""
		# Use Transformer Positional Encoding to get time embeddings
		freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).cuda(self.device) # size = (5)
		temp = timesteps.unsqueeze(-1).float() * freqs.unsqueeze(0)  # (batch, 5)
		time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)  # (batch, 10)
		if self.time_emb_dim % 2:
			# Force the last dimension is even (cat with zeros col)
			time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
		emb = self.emb_layer(time_emb)  # (batch, 10)

		if self.norm:
			x_t = F.normalize(x_t)
		if mess_dropout:
			x_t = self.drop(x_t)
		
		h = torch.cat([x_t, emb], dim=-1)  # (batch, item_num+10)
		for i, layer in enumerate(self.in_layers):
			h = layer(h)
			h = torch.tanh(h)  #? how about other activation functions?
		for i, layer in enumerate(self.out_layers):
			h = layer(h)
			if i != len(self.out_layers) - 1:
				# not the last layer
				h = torch.tanh(h)

		return h  # (batch, item_num)

class GaussianDiffusion(nn.Module):
	def __init__(self, config: Config, beta_fixed=True):
		super(GaussianDiffusion, self).__init__()
		self.device = torch.device(f"cuda:{config.base.gpu}" if torch.cuda.is_available() else "cpu")
		self.noise_scale = config.hyper.noise_scale
		self.noise_min = config.hyper.noise_min
		self.noise_max = config.hyper.noise_max
		self.steps = config.hyper.steps

		if self.noise_scale != 0:
			self.betas = torch.tensor(self.get_betas(), dtype=torch.float64, device=self.device)
			if beta_fixed:
				self.betas[0] = 0.0001

			self.calculate_for_diffusion()

	def get_betas(self):
		"""自适应生成 Beta 噪声"""
		start = self.noise_scale * self.noise_min
		end = self.noise_scale * self.noise_max
		variance = np.linspace(start, end, self.steps, dtype=np.float64)

		# 自适应调整 Beta
		alpha_bar = 1 - variance
		betas = []
		betas.append(1 - alpha_bar[0])
		for i in range(1, self.steps):
			adaptive_factor = np.log(1 + i) / np.log(1 + self.steps)  # 自适应因子
			beta = min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999) * adaptive_factor
			betas.append(beta)
			# betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
		return np.array(betas)

	def calculate_for_diffusion(self):
		alphas = 1.0 - self.betas  # α = 1-β: (5)
		self.alphas_cumprod = torch.cumprod(alphas, dim=0)  # \bar{α}: (5)
		self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]])  # t-1 step
		self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0], device=self.device)])   # t+1 step

		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)  # \sqrt{\bar{α}}
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)  # \sqrt{1-\bar{α}}
		self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)  # \log(1-\bar{α})
		self.sqrt_reciprocal_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)  # \sqrt{ 1 / \bar{α} }
		self.sqrt_reciprocalm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)  # \sqrt{ 1/\bar{α} - 1}
		
		# predicted β = β x (1-\bar{α}_{t-1}) / (1-\bar{α}_{t})
		self.posterior_variance = (
			self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		#! because pos_var[0] = 0, so we need to clip it. use pos_var[1] instead
		#? so is it because of the determined beta?
		self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
		# x0's coefficient = β x \sqrt{\bar{α}_{t-1}} / (1-\bar{α}_{t})
		#! for inverse diffusion process (denoise), x0 is a unknown variable. is this correct?
		self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
		# xt's coefficient = (1-\bar{α}_{t-1}) x \sqrt{α} / (1-\bar{α}_{t})
		self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

	def calculate_for_diffusion2(self):
		alphas = 1.0 - self.betas  # α = 1 - β
		self.alphas_cumprod = torch.cumprod(alphas, dim=0)  # \bar{α}
		self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]])  # t-1 step
		self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0], device=self.device)])   # t+1 step

		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)  # \sqrt{\bar{α}}
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)  # \sqrt{1 - \bar{α}}

		# 后验方差
		self.posterior_variance = (
			self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		self.posterior_log_variance_clipped = torch.log(
			torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
		)

		# 均值计算简化
		self.posterior_mean_coef1 = 1.0 / torch.sqrt(alphas)  # 1 / sqrt(α)
		self.posterior_mean_coef2 = (1.0 - alphas) / (torch.sqrt(1.0 - self.alphas_cumprod))  # (1 - α) / sqrt(1 - \bar{α})

	def backward_steps(self, model: Denoise, x_start: torch.Tensor, sampling_steps: int, sampling_noise=False):
		"""
		Implement inverse diffusion process (sampling start from `sampling_steps`)

		Args:
			model (Denoise): use for calculate posterior mean and posterior variance
			x_start (torch.Tensor): (batch_size, item_num)
			sampling_steps (int): sampling start
			sampling_noise (bool): whether to add noise
		"""
		#todo: check sampling_steps > config.train.steps?
		if sampling_steps == 0:
			x_t = x_start  #! steps default = 0
		else:
			timesteps = torch.tensor([sampling_steps-1] * x_start.shape[0], device=self.device)
			x_t = self.forward_cal_xt(x_start, timesteps)
		
		indices = list(range(self.steps))[::-1]  # reverse order

		x0 = x_t # init
		for i in indices:
			timesteps = torch.tensor([i] * x_t.shape[0], device=self.device)  # (ttt...batch_size)
			model_mean, model_log_variance = self.p_mean_variance(model, x_t, timesteps)
			if sampling_noise:
				noise = torch.randn_like(x_t)
				nonzero_mask = ((timesteps!=0).float().view(-1, *([1]*(len(x_t.shape)-1))))
				x0 = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
			else:
				x0 = model_mean
		return x0

	def forward_cal_xt(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: Optional[torch.Tensor] = None):
		"""
		Calculate x_t from x_0 and noise.

		Args:
			x_start: (batch, item)
			timesteps: (batch,)
			noise: Default is None. If not provided, torch.randn_like(x_start)

		Returns:
			torch.Tensor: Forward diffusion process x_t.
		"""
		if noise is None:
			noise = torch.randn_like(x_start)
		# x_t = \sqrt{\bar{α}_{t}} * x_0 + \sqrt{1-\bar{α}_{t}} * noise
		x0_coef = self._extract_into_tensor(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
		noise_coef = self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape)
		return x0_coef * x_start + noise_coef * noise

	def _extract_into_tensor(self, var: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: torch.Size):
		"""
		Extract every timestep's posterior variance, and broadcast it to the batch_item.
		
		Args:
			var (torch.Tensor): posterior_variance (5)
			timesteps (torch.Tensor): (ttt...batch)
			broadcast_shape (torch.Size): (batch, item)
		"""
		res = var[timesteps].float() #? .float() is necessary?
		while len(res.shape) < len(broadcast_shape):
			res = res.unsqueeze(-1)
		return res.expand(broadcast_shape)

	def p_mean_variance(self, denoise: Denoise, x_t: torch.Tensor, timesteps: torch.Tensor):
		"""
		calculate posterior mean and posterior variance for inverse diffusion process

		Args:
			x (torch.Tensor): (batch_size, item_num)
			timesteps (torch.Tensor): (ttt...batch_size)
		"""
		predicted_alpha0 = denoise(x_t, timesteps, False)

		model_variance = self.posterior_variance
		model_log_variance = self.posterior_log_variance_clipped

		model_variance = self._extract_into_tensor(model_variance, timesteps, x_t.shape)
		model_log_variance = self._extract_into_tensor(model_log_variance, timesteps, x_t.shape)

		model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, timesteps, x_t.shape) * predicted_alpha0 + self._extract_into_tensor(self.posterior_mean_coef2, timesteps, x_t.shape) * x_t)
		
		return model_mean, model_log_variance

	def calculate_time_step_weights(self):
		"""
		Calculate weights for each time step based on SNR or other metrics.
		Returns:
			torch.Tensor: Weights for each time step (steps,).
		"""
		# 使用 SNR 作为时间步的重要性指标
		snr = self.alphas_cumprod / (1 - self.alphas_cumprod + 1e-8)  # SNR
		weights = torch.sqrt(snr)  # 对 SNR 取平方根，增强对高 SNR 时间步的偏好
		return weights / weights.sum()  # 归一化为概率分布

	def sample_timesteps(self, batch_size: int):
		"""
		Sample time steps for a batch using non-uniform sampling.
		Args:
			batch_size (int): Number of samples in the batch.
		Returns:
			torch.Tensor: Sampled time steps (batch_size,).
		"""
		weights = self.calculate_time_step_weights()  # 获取时间步的采样权重
		timesteps = torch.multinomial(weights, batch_size, replacement=True)  # 根据权重采样
		return timesteps

	def training_losses(self, model: Denoise, x_start: torch.Tensor, i_embs: torch.Tensor, model_feats: torch.Tensor):
		"""
		Args:
			x_start: (batch, item)
			i_embs: (item, dim)
			model_feats: (item, dim)
		
		Returns:
			tuple (Tensor, Tensor): (fit_noise_loss, refact_ui_loss): (batch, batch)
		"""
		batch_size = x_start.size(0)

		# 使用非均匀采样策略选择时间步
		timesteps = self.sample_timesteps(batch_size)
		# timesteps = torch.randint(0, self.steps, (batch_size,), device=self.device)

		noise = torch.randn_like(x_start)
		if self.noise_scale != 0:
			# forward diffusion
			x_t = self.forward_cal_xt(x_start, timesteps, noise)
		else:
			x_t = x_start
		# backward diffusion (denoise)
		model_output = model.forward(x_t, timesteps)

		mse_loss = self.MSELoss(x_start, model_output)  # (batch)
		weight = self.SNR(timesteps - 1) - self.SNR(timesteps)  # the \lambda_0 in paper
		weight = torch.where((timesteps == 0), 1.0, weight)
		fit_noise_loss = weight * mse_loss  # ELBO loss

		user_modal_embs = torch.mm(model_output, model_feats)  # (batch, dim)
		user_id_embs = torch.mm(x_start, i_embs)  # (batch, dim)
		refact_ui_loss = self.MSELoss(user_modal_embs, user_id_embs)  # (batch)

		return fit_noise_loss, refact_ui_loss

	def MSELoss(self, x_start: torch.Tensor, model_output: torch.Tensor):
		"""
		Compute the batch MSE loss between the origin view and denoised view.

		Returns:
			torch.Tensor: (batch,)
		"""
		dim_except_batch = list(range(1, len(x_start.shape)))
		mse = torch.mean((x_start - model_output) ** 2, dim=dim_except_batch)  # (batch, )
		return mse
	
	def SNR(self, t: torch.Tensor) -> torch.Tensor:
		"""Compute the Signal-to-Noise Ratio (SNR) at a given timestep."""
		epsilon = 1e-8
		return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t] + epsilon)