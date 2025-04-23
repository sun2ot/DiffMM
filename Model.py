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
		self.u_embs = nn.Parameter(init(torch.empty(self.config.data.user_num, self.config.base.latdim))) # type: ignore
		self.i_embs = nn.Parameter(init(torch.empty(self.config.data.item_num, self.config.base.latdim))) # type: ignore

		if self.config.base.trans == 1:
			self.image_layer = nn.Linear(self.config.data.image_feat_dim, self.config.base.latdim)
			self.text_layer = nn.Linear(self.config.data.text_feat_dim, self.config.base.latdim)
		elif self.config.base.trans == 0:
			self.image_matrix = nn.Parameter(init(torch.empty(size=(self.config.data.image_feat_dim, self.config.base.latdim)))) # type: ignore
			self.text_matrix = nn.Parameter(init(torch.empty(size=(self.config.data.text_feat_dim, self.config.base.latdim)))) # type: ignore
		else:  # self.config.base.trans == 2
			self.image_matrix = nn.Parameter(init(torch.empty(size=(self.config.data.image_feat_dim, self.config.base.latdim)))) # type: ignore
			self.text_layer = nn.Linear(self.config.data.text_feat_dim, self.config.base.latdim)
		if audio_embedding is not None:
			if self.config.base.trans == 1:
				self.audio_layer = nn.Linear(self.config.data.audio_feat_dim, self.config.base.latdim)
			else:
				self.audio_matrix = nn.Parameter(init(torch.empty(size=(self.config.data.audio_feat_dim, self.config.base.latdim)))) # type: ignore

		self.image_embedding = image_embedding
		self.text_embedding = text_embedding
		if audio_embedding is not None:
			self.audio_embedding = audio_embedding
		else:
			self.audio_embedding = None
		
		# average weight
		if audio_embedding is not None:
			self.modal_weight = nn.Parameter(torch.tensor([0.3333, 0.3333, 0.3333])) # type: ignore
		else:
			self.modal_weight = nn.Parameter(torch.tensor([0.5, 0.5])) # type: ignore
		self.softmax = nn.Softmax(dim=-1)
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

	def getAudioFeats(self) -> Optional[torch.Tensor]:
		if self.audio_embedding is None:
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

		if audio_adj is not None:
			if self.config.base.trans == 0:
				assert self.audio_embedding is not None
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_matrix))
			else:
				audio_feats = self.audio_layer(self.audio_embedding)
		else:
			audio_feats = None

		weight: nn.Parameter = self.softmax(self.modal_weight) # type: ignore

		# image
		image_adj_embs = torch.cat([self.u_embs, F.normalize(image_feats)])  # (node, dim)
		image_adj_embs = torch.sparse.mm(image_adj, image_adj_embs)  # (node, dim) #! 这个就是gcn_MM_CL的返回值

		image_aware_embs = torch.cat([self.u_embs, self.i_embs])  # (node, dim)
		image_aware_embs = torch.sparse.mm(adj, image_aware_embs)  # (node, dim)
		
		# text
		text_adj_embs = torch.cat([self.u_embs, F.normalize(text_feats)])
		text_adj_embs = torch.sparse.mm(text_adj, text_adj_embs)

		text_aware_embs = torch.cat([self.u_embs, self.i_embs])
		text_aware_embs = torch.sparse.mm(adj, text_aware_embs)

		# audio
		if audio_adj is not None:
			assert audio_feats is not None
			audio_adj_embs = torch.cat([self.u_embs, F.normalize(audio_feats)])
			audio_adj_embs = torch.sparse.mm(audio_adj, audio_adj_embs)

			audio_aware_embs = torch.cat([self.u_embs, self.i_embs])
			audio_aware_embs = torch.sparse.mm(adj, audio_aware_embs)

		else:
			audio_adj_embs, audio_aware_embs = None, None

		image_aware_embs += self.config.hyper.modal_adj_weight * image_adj_embs
		text_aware_embs += self.config.hyper.modal_adj_weight * text_adj_embs
		if audio_adj is not None:
			assert audio_adj_embs is not None
			audio_aware_embs += self.config.hyper.modal_adj_weight * audio_adj_embs
		
		if audio_adj is None:
			modal_embs = weight[0] * image_aware_embs + weight[1] * text_aware_embs
		else:
			modal_embs = weight[0] * image_aware_embs + weight[1] * text_aware_embs + weight[2] * audio_aware_embs

		final_embs = modal_embs
		embs_list = [final_embs]
		# 只做一层GCN
		final_embs = torch.sparse.mm(adj, embs_list[-1])
		embs_list.append(final_embs)
		final_embs = torch.stack(embs_list).sum(dim=0)
		final_embs = final_embs + self.config.hyper.residual_weight * modal_embs

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

		if audio_adj is not None:
			if self.config.base.trans == 0:
				assert self.audio_embedding is not None
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_matrix))
			else:
				audio_feats = self.audio_layer(self.audio_embedding)
		else:
			audio_feats = None

		image_aware_embs = torch.cat([self.u_embs, F.normalize(image_feats)])
		image_aware_embs = torch.sparse.mm(image_adj, image_aware_embs)

		text_aware_embs = torch.cat([self.u_embs, F.normalize(text_feats)])
		text_aware_embs = torch.sparse.mm(text_adj, text_aware_embs)

		if audio_adj is not None:
			assert audio_feats is not None
			audio_aware_embs = torch.cat([self.u_embs, F.normalize(audio_feats)])
			audio_aware_embs = torch.sparse.mm(audio_adj, audio_aware_embs)
		else:
			audio_aware_embs = None

		image_modal_embs = image_aware_embs
		text_modal_embs = text_aware_embs

		if audio_adj is not None:
			assert audio_aware_embs is not None
			audio_modal_embs = audio_aware_embs
		else:
			audio_modal_embs = None

		if audio_adj is None:
			return image_modal_embs[:self.config.data.user_num], image_modal_embs[self.config.data.user_num:], text_modal_embs[:self.config.data.user_num], text_modal_embs[self.config.data.user_num:]
		else:
			assert audio_modal_embs is not None
			return image_modal_embs[:self.config.data.user_num], image_modal_embs[self.config.data.user_num:], text_modal_embs[:self.config.data.user_num], text_modal_embs[self.config.data.user_num:], audio_modal_embs[:self.config.data.user_num], audio_modal_embs[self.config.data.user_num:]
		
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

		self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim) # (10, 10)

		in_dims_temp: list = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]  # (item_num+10, 1000)
		out_dims_temp: list = self.out_dims  # (1000, item_num)

		# (item_num+10, 1000)
		self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
		# (1000, item_num)
		self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

		self.drop = nn.Dropout(dropout)
		self.init_weights()

		self.modal_emb_dim = config.base.latdim  # 单一模态特征维度
		self.gate_layer = nn.Linear(self.modal_emb_dim, self.modal_emb_dim)  # 门控机制，输出与模态特征维度一致

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

	def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor, modal_feat: Optional[Tensor] = None) -> torch.Tensor:
		"""
		Denoise Layer

		Args:
			x_t (torch.Tensor): batch_u_items, (batch, item)
			timesteps (torch.Tensor): (batch,)
			modal_feat (Tensor): 模态特征 (item_num, latdim)
		
		Returns:
			h (torch.Tensor): denoised view (batch, item)
		"""
		# Use Transformer Positional Encoding to get time embeddings
		freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32).to(self.device) / (self.time_emb_dim//2))
		temp = timesteps.unsqueeze(-1).float() * freqs.unsqueeze(0)  # (batch, 5)
		time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)  # (batch, 10)
		if self.time_emb_dim % 2:
			# Force the last dimension is even (cat with zeros col)
			time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
		time_emb = self.emb_layer(time_emb)  # (batch, 10)

		if modal_feat is not None:
			modal_feat_projected = torch.mm(x_t, modal_feat)  # (batch, latdim)
			modal_weights = torch.sigmoid(self.gate_layer(modal_feat_projected))  # (batch, latdim)
			modal_feats_adjusted = modal_feat_projected * modal_weights
			x_t = x_t + torch.mm(modal_feats_adjusted, modal_feat.T)
		
		h = torch.cat([x_t, time_emb], dim=-1)  # (batch, item_num+10)
		for i, layer in enumerate(self.in_layers):
			h = layer(h)
			h = torch.tanh(h)
		for i, layer in enumerate(self.out_layers):
			h = layer(h)
			if i != len(self.out_layers) - 1:
				# not the last layer
				h = torch.tanh(h)

		return h  # (batch, item)

class GaussianDiffusion(nn.Module):
	def __init__(self, config: Config, beta_fixed=True):
		super(GaussianDiffusion, self).__init__()
		self.config = config
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
		start = self.noise_scale * self.noise_min
		end = self.noise_scale * self.noise_max
		variance = np.linspace(start, end, self.steps, dtype=np.float64)

		alpha_bar = 1 - variance
		betas = []
		betas.append(1 - alpha_bar[0])
		for i in range(1, self.steps):
			beta = min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999)
			betas.append(beta)
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
		if sampling_steps == 0:
			x_t = x_start  #! steps default = 0
		else:
			timesteps = torch.tensor([sampling_steps-1] * x_start.shape[0], device=self.device)
			x_t = self.forward_cal_xt(x_start, timesteps)
		
		indices = list(range(self.steps))[::-1]  # reverse order

		for i in indices:
			timesteps = torch.tensor([i] * x_t.shape[0], device=self.device) # Select the most probable time step
			model_mean, _ = self.p_mean_variance(model, x_t, timesteps)
			x_t = model_mean
		return x_t

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
			noise = torch.sign(x_start) * F.normalize(torch.randn_like(x_start))
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
		res = var[timesteps].float()
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
		predicted_alpha0 = denoise.forward(x_t, timesteps)

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
	
	def SNR(self, t: torch.Tensor) -> torch.Tensor:
		"""Compute the Signal-to-Noise Ratio (SNR) at a given timestep."""
		epsilon = 1e-8
		return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t] + epsilon)

	def training_losses(self, model: Denoise, x_start: torch.Tensor, i_embs: torch.Tensor, modal_feat: torch.Tensor):
		"""
		Args:
			x_start: (batch, item)
			i_embs: (item, dim)
			modal_feat: (item_num, latdim)
		
		Returns:
			torch.Tensor: total_loss
		"""
		batch_size = x_start.size(0)

		# 使用非均匀采样策略选择时间步
		timesteps = self.sample_timesteps(batch_size)

		# 添加噪声
		noise = torch.randn_like(x_start)
		if self.noise_scale != 0:
			x_t = self.forward_cal_xt(x_start, timesteps, noise)  # forward diffusion
		else:
			x_t = x_start

		# 去噪过程
		model_output = model.forward(x_t, timesteps, modal_feat=modal_feat)  # (batch, item)

		# 1. 重构损失 (Reconstruction Loss)
		reconstruction_loss = F.mse_loss(model_output, x_start, reduction='none')  # (batch, item)
		reconstruction_loss = reconstruction_loss.mean(dim=-1)  # (batch,)
		# 防止timesteps-1为负
		timesteps_minus1 = torch.clamp(timesteps - 1, min=0)
		weight = self.SNR(timesteps_minus1) - self.SNR(timesteps)  # the \lambda_0 in paper
		weight = torch.where((timesteps == 0), torch.tensor(1.0, device=weight.device), weight)
		reconstruction_loss = weight * reconstruction_loss  # (batch,)

		# 2. 对比损失 (Contrastive Loss)
		user_modal_embs = torch.sparse.mm(model_output, modal_feat)  # (batch, latdim)
		user_id_embs = torch.sparse.mm(x_start, i_embs)  # (batch, latdim)
		contrastive_loss = 1 - F.cosine_similarity(user_modal_embs, user_id_embs, dim=-1)  # (batch,)

		# 3. 正则化损失 (Regularization Loss)
		reg_loss = l2_reg_loss(self.config.train.reg, [i_embs], self.device)  # 标量
		reg_loss = reg_loss.expand(batch_size)  # (batch,)

		# 4. 动态权重平衡
		total_loss = reconstruction_loss + contrastive_loss * self.config.hyper.e_loss + reg_loss * self.config.train.reg   # (batch,)

		return total_loss