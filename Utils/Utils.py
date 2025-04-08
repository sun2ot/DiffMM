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