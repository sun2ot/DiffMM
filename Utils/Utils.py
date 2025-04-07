import torch
import torch.nn.functional as F
from torch import Tensor

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
    计算L2正则化损失

    Args:
        reg (float): 正则化系数
        embeddings (List[Tensor]): 需要进行正则化的张量列表
    
    Returns:
        正则化损失的平均值，标量张量
    """
    emb_loss = torch.tensor(0., device=device)
    for emb in embeddings:
        # 计算每一个嵌入向量的L2范数，然后除以batch_size以减少其对损失函数的影响
        # 累加到总的嵌入损失中
        # emb_loss += torch.linalg.vector_norm(emb, ord=2)
        # emb_loss += 1./2*torch.sum(emb**2)
        emb_loss += torch.norm(emb, p=2).square()
    return emb_loss * reg


def InfoNCE(batch_view1: torch.Tensor, batch_view2: torch.Tensor, idx: Tensor, temperature: float, b_cos: bool = True):
    """
    计算InfoNCE损失函数

    Args:
        view1 (torch.Tensor): Num x Dim
        view2 (torch.Tensor): Num x Dim
        temperature (float): 温度系数
        b_cos (bool): 是否使用余弦相似度

    Returns:
        Average InfoNCE Loss
    """
    batch_view1 = batch_view1[idx]
    batch_view2 = batch_view2[idx]
    if batch_view1.shape != batch_view2.shape:
        raise ValueError(f"InfoNCE expected the same shape for two views. But got view1.shape={batch_view1.shape} and view2.shape={batch_view2.shape}.")
    # 如果使用余弦相似度，则先进行归一化
    if b_cos:
        batch_view1, batch_view2 = F.normalize(batch_view1, p=2, dim=1), F.normalize(batch_view2, p=2, dim=1)
    # 计算正样本的分数，使用点积并除以温度参数
    pos_score = (batch_view1 @ batch_view2.T) / temperature
    # 计算每个样本的分数
    score = torch.diag(F.log_softmax(pos_score, dim=1))
    return -score.mean()