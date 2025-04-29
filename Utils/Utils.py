import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from numba import njit, prange


def innerProduct(user_embs, item_embs) -> Tensor:
	return torch.sum(user_embs * item_embs, dim=-1)

def pairPredict(u_embs, pos_embs, neg_embs) -> Tensor:
	"""positive pair inner product - negative pair inner product"""
	return innerProduct(u_embs, pos_embs) - innerProduct(u_embs, neg_embs)

def calcRegLoss(model):
	ret = 0
	for W in model.parameters():
		ret += W.norm(2).square()
	return ret

def calcReward(bprLossDiff, keepRate):
	_, posLocs = torch.topk(bprLossDiff, int(bprLossDiff.shape[0] * (1 - keepRate)))
	reward = torch.zeros_like(bprLossDiff).cuda()
	reward[posLocs] = 1.0
	return reward

def calcGradNorm(model):
	ret = 0
	for p in model.parameters():
		if p.grad is not None:
			ret += p.grad.data.norm(2).square()
	ret = (ret ** 0.5)
	ret.detach()
	return ret

def contrastLoss(embeds1, embeds2, nodes, temp):
	embeds1 = F.normalize(embeds1, p=2)
	embeds2 = F.normalize(embeds2, p=2)
	pckEmbeds1 = embeds1[nodes]
	pckEmbeds2 = embeds2[nodes]
	nume = torch.exp(torch.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
	deno = torch.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1)
	return -torch.log(nume / deno).mean()

def l2_reg_loss(reg: float, embeddings: list[Tensor], device: torch.device) -> Tensor:
    """
    Args:
        reg (float): reg weight
        embeddings (List[Tensor]): List of embeddings to be regularized
    """
    emb_loss = torch.tensor(0., device=device)
    for emb in embeddings:
        emb_loss += torch.sum(emb**2)
    return emb_loss * reg


def InfoNCE(batch_view1: Tensor, batch_view2: Tensor, idx: Tensor, temperature: float, b_cos: bool = True):
    """
    Args:
        view1 (Tensor): Num x Dim
        view2 (Tensor): Num x Dim
        b_cos (bool): Whether to use cosine similarity

    Returns:
        Average InfoNCE Loss
    """
    batch_view1 = batch_view1[idx]
    batch_view2 = batch_view2[idx]
    if batch_view1.shape != batch_view2.shape:
        raise ValueError(f"InfoNCE expected the same shape for two views. But got view1.shape={batch_view1.shape} and view2.shape={batch_view2.shape}.")
    if b_cos:
        batch_view1, batch_view2 = F.normalize(batch_view1, p=2, dim=1), F.normalize(batch_view2, p=2, dim=1)
    pos_score = (batch_view1 @ batch_view2.T) / temperature
    score = torch.diag(F.log_softmax(pos_score, dim=1))
    return -score.mean()


def bpr_loss(user_emb, pos_item_emb, neg_item_embs):
    """
    计算Bayesian Personalized Ranking (BPR) 损失函数
    
    Args:
        user_emb: 用户嵌入向量，形状为[batch_size, embedding_dim]
        pos_item_emb: 正样本物品嵌入向量，形状为[batch_size, embedding_dim]
        neg_item_emb: 负样本物品嵌入向量，形状为[batch_size, embedding_dim]
    
    Returns:
        BPR损失的平均值, 标量张量
    """
    # 计算用户对正样本的偏好分数
    # torch.mul is element-wise multiplies
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)  # (batch_size)
    # 计算用户对负样本的偏好分数
    neg_score = torch.mul(user_emb, neg_item_embs).sum(dim=1)  # (batch_size)
    # BPR损失: 对每个正样本，计算它与所有负样本的损失
    loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))  # (batch_size)
    # 返回损失的均值
    return torch.mean(loss)


@njit
def l2_norm(x):
    s = 0.0
    for i in range(x.shape[0]):
        s += x[i] * x[i]
    return np.sqrt(s)

@njit(parallel=True)
def compute_user_protos(user_pos_items, item_feats, user_proto):
    """
    user_pos_items: 2D object array/list, 每行是该用户历史交互的 item 索引 array
    item_feats: (item_num, feat_dim) float32
    user_proto: (user_num, feat_dim) pre分配好的 output
    """
    U, D = user_proto.shape
    for u in prange(U):
        items = user_pos_items[u]
        if items.size == 0:
            continue
        # sum pooling
        tmp = np.zeros(D, dtype=np.float32)
        for idx in items:
            for d in range(D):
                tmp[d] += item_feats[idx, d]
        # 平均
        for d in range(D):
            user_proto[u, d] = tmp[d] / items.size

@njit(parallel=True)
def build_knn_adj_numba(user_pos_items, item_feats, topk, u_list, i_list, vals):
    """
    输出边表 (u_list, i_list, vals)，长度 = user_num * topk
    """
    U, D = user_pos_items.shape[0], item_feats.shape[1]
    N = item_feats.shape[0]
    # 1. 先算每个用户的原型
    user_proto = np.zeros((U, D), dtype=np.float32)
    compute_user_protos(user_pos_items, item_feats, user_proto)

    # 2. 逐用户并行算相似度 & top-k
    for u in prange(U):
        # 先算 user_proto[u] norm
        un = l2_norm(user_proto[u])
        # 逐 item 算余弦（dot 除以 norm）
        sim = np.empty(N, dtype=np.float32)
        for j in range(N):
            dot = 0.0
            for d in range(D):
                dot += user_proto[u, d] * item_feats[j, d]
            dn = l2_norm(item_feats[j])
            sim[j] = dot / (un * dn + 1e-8)
        # 简单排序取 topk
        idx_sorted = np.argsort(sim)
        base = u * topk
        for k in range(topk):
            # 取最大的 topk
            ii = idx_sorted[N - 1 - k]
            u_list[base + k] = u
            i_list[base + k] = ii
            vals[base + k]   = 1.0
