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

		self.image_layer = nn.Linear(self.config.data.image_feat_dim, self.config.base.latdim)
		self.text_layer = nn.Linear(self.config.data.text_feat_dim, self.config.base.latdim)
		if audio_embedding is not None:
			self.audio_layer = nn.Linear(self.config.data.audio_feat_dim, self.config.base.latdim)

		self.image_embedding = image_embedding
		self.text_embedding = text_embedding
		self.audio_embedding = audio_embedding
		
		# average weight
		if audio_embedding is not None:
			self.modal_weight = nn.Parameter(torch.tensor([0.3333, 0.3333, 0.3333])) # type: ignore
		else:
			self.modal_weight = nn.Parameter(torch.tensor([0.5, 0.5])) # type: ignore
		self.softmax = nn.Softmax(dim=-1)
		self.leakyrelu = nn.LeakyReLU(0.2)
				
	def getItemEmbs(self):
		return self.i_embs
	
	def getUserEmbs(self):
		return self.u_embs
	
	def getImageFeats(self) -> torch.Tensor:
		return self.image_layer(self.image_embedding)
	
	def getTextFeats(self) -> torch.Tensor:
		return self.text_layer(self.text_embedding)

	def getAudioFeats(self) -> Optional[torch.Tensor]:
		if self.audio_embedding is None:
			return None
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
		@dataclass
		class GCNOutput:
			u_final_embs: Tensor
			i_final_embs: Tensor
			u_image_embs: Tensor
			i_image_embs: Tensor
			u_text_embs: Tensor
			i_text_embs: Tensor
			u_audio_embs: Optional[Tensor] = None
			i_audio_embs: Optional[Tensor] = None

		 # Trans multimodal feats to 64 (latdim)
		image_feats = self.image_layer(self.image_embedding)
		text_feats = self.text_layer(self.text_embedding)

		weight: nn.Parameter = self.softmax(self.modal_weight) # type: ignore

		image_adj_embs = torch.cat([self.u_embs, F.normalize(image_feats)])  # (node, dim)
		image_adj_embs = torch.sparse.mm(image_adj, image_adj_embs)  # (node, dim)

		text_adj_embs = torch.cat([self.u_embs, F.normalize(text_feats)])
		text_adj_embs = torch.sparse.mm(text_adj, text_adj_embs)

		user = self.config.data.user_num
		gcn_output = GCNOutput(
			image_adj_embs[:user], image_adj_embs[user:],  # just u/i_final_embs placeholder
			image_adj_embs[:user], image_adj_embs[user:],
			text_adj_embs[:user], text_adj_embs[user:]
		)

		if self.audio_embedding is not None:
			audio_feats = self.audio_layer(self.audio_embedding)
			audio_adj_embs = torch.cat([self.u_embs, F.normalize(audio_feats)])
			audio_adj_embs = torch.sparse.mm(audio_adj, audio_adj_embs)
			gcn_output.u_audio_embs, gcn_output.i_audio_embs = audio_adj_embs[:user], audio_adj_embs[user:]
		else:
			audio_adj_embs = None

		image_aware_embs = torch.cat([self.u_embs, self.i_embs])  # (node, dim)
		image_aware_embs = torch.sparse.mm(adj, image_aware_embs)  # (node, dim)

		text_aware_embs = torch.cat([self.u_embs, self.i_embs])
		text_aware_embs = torch.sparse.mm(adj, text_aware_embs)

		image_aware_embs += self.config.hyper.modal_adj_weight * image_adj_embs
		text_aware_embs += self.config.hyper.modal_adj_weight * text_adj_embs

		modal_embs = weight[0] * image_aware_embs + weight[1] * text_aware_embs

		if audio_adj_embs is not None:
			audio_aware_embs = torch.cat([self.u_embs, self.i_embs])
			audio_aware_embs = torch.sparse.mm(adj, audio_aware_embs)

			audio_aware_embs += self.config.hyper.modal_adj_weight * audio_adj_embs
			
			modal_embs += weight[2] * audio_aware_embs

		final_embs = modal_embs
		final_embs += torch.sparse.mm(adj, modal_embs)
		final_embs += self.config.hyper.residual_weight * modal_embs
		gcn_output.u_final_embs, gcn_output.i_final_embs = final_embs[:self.config.data.user_num], final_embs[self.config.data.user_num:]

		return gcn_output

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

		# 计算简化
		# 1 / sqrt(α)
		self.posterior_mean_coef1 = 1.0 / torch.sqrt(alphas)
		# (1 - α) / sqrt(1 - \bar{α})
		self.posterior_mean_coef2 = (1.0 - alphas) / torch.sqrt(1.0 - self.alphas_cumprod)

	def generate_view(self, model: Denoise, x_start: torch.Tensor, sampling_step: int):
		"""
		Implement inverse diffusion process (sampling start from `sampling_steps`)

		Args:
			model (Denoise): use for calculate posterior mean and posterior variance
			x_start (torch.Tensor): (batch_size, item_num)
			sampling_steps (int): sampling start
			sampling_noise (bool): whether to add noise
		"""
		if sampling_step == 0:
			x_t = x_start  #! steps default = 0
		else:
			timesteps = torch.tensor([sampling_step-1] * x_start.shape[0], device=self.device)
			x_t = self.forward_cal_xt(x_start, timesteps)
		
		indices = list(range(self.steps))[::-1]  # reverse order

		for i in indices:
			timesteps = torch.tensor([i] * x_t.shape[0], device=self.device) # Select the most probable time step
			model_mean, _ = self.p_mean_variance(model, x_t, timesteps)
			x_t = model_mean
		return x_t

	def forward_cal_xt(self, x_0: torch.Tensor, timesteps: torch.Tensor, noise: Optional[torch.Tensor] = None):
		"""
		Calculate x_t from x_0 and noise.

		Args:
			x_0: (batch, item)
			timesteps: (batch,)
			noise: Default is None. If not provided, torch.randn_like(x_0)

		Returns:
			torch.Tensor: Forward diffusion process x_t.
		"""
		if noise is None:
			noise = torch.sign(x_0) * F.normalize(torch.randn_like(x_0))
		# x_t = \sqrt{\bar{α}_{t}} * x_0 + \sqrt{1-\bar{α}_{t}} * noise
		x0_coef = self._extract_into_tensor(self.sqrt_alphas_cumprod, timesteps, x_0.shape)
		noise_coef = self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, timesteps, x_0.shape)
		return x0_coef * x_0 + noise_coef * noise

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

	def p_mean_variance(self, denoise: Denoise, x_t: torch.Tensor, timesteps: torch.Tensor, noise: Optional[Tensor] = None):
		"""
		calculate posterior mean and posterior variance for inverse diffusion process

		Args:
			x (torch.Tensor): (batch_size, item_num)
			timesteps (torch.Tensor): (ttt...batch_size)
		"""
		predicted_x0 = denoise.forward(x_t, timesteps)
		# if noise is None:
		# 	noise = torch.randn_like(x_t)

		model_variance = self.posterior_variance
		model_log_variance = self.posterior_log_variance_clipped

		model_variance = self._extract_into_tensor(model_variance, timesteps, x_t.shape)
		model_log_variance = self._extract_into_tensor(model_log_variance, timesteps, x_t.shape)

		model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, timesteps, x_t.shape) * predicted_x0 + self._extract_into_tensor(self.posterior_mean_coef2, timesteps, x_t.shape) * x_t)
		# model_mean = self._extract_into_tensor(self.posterior_mean_coef1, timesteps, x_t.shape) * (x_t - self._extract_into_tensor(self.posterior_mean_coef2, timesteps, x_t.shape) * noise)
		
		return model_mean, model_log_variance
	
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

		timesteps = torch.randint(0, self.steps, (batch_size,)).long().cuda()

		# 添加噪声
		noise = torch.randn_like(x_start)
		x_t = self.forward_cal_xt(x_start, timesteps, noise)  # forward diffusion

		# 去噪过程
		model_output = model.forward(x_t, timesteps, modal_feat=modal_feat)  # (batch, item)

		# 重构损失 (Reconstruction Loss)
		reconstruction_loss = F.mse_loss(model_output, x_start, reduction='none')  # (batch, item)
		reconstruction_loss = reconstruction_loss.mean(dim=-1)  # (batch,)
		# 防止timesteps-1为负
		timesteps_minus1 = torch.clamp(timesteps - 1, min=0)
		weight = self.SNR(timesteps_minus1) - self.SNR(timesteps)
		weight = torch.where((timesteps == 0), torch.tensor(1.0, device=weight.device), weight)
		reconstruction_loss = weight * reconstruction_loss  # (batch,)

		# 偏好相似度损失 (Preference Similarity Loss)
		user_modal_embs = torch.sparse.mm(model_output, modal_feat)  # (batch, latdim)
		user_id_embs = torch.sparse.mm(x_start, i_embs)  # (batch, latdim)
		sim_loss = 1 - F.cosine_similarity(user_modal_embs, user_id_embs, dim=-1)  # (batch,)

		# 正则化损失 (Regularization Loss)
		reg_loss = l2_reg_loss(self.config.train.reg, [i_embs], self.device)  # 标量
		reg_loss = reg_loss.expand(batch_size)  # (batch,)

		# 动态权重平衡
		total_loss = reconstruction_loss + sim_loss * self.config.hyper.sim_weight + reg_loss * self.config.train.reg   # (batch,)
		# total_loss = reconstruction_loss + reg_loss * self.config.train.reg   # (batch,)

		return total_loss