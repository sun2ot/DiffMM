import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix
from Params import args
import torch
from torch.utils.data import Dataset as torch_dataset
import torch.utils.data as dataloader
from collections import defaultdict
from tqdm import tqdm
import random
from typing import Optional

class DataHandler:
	def __init__(self):
		"""
		Initialize dataset's path based on the dataset name from `args.data`.
		"""
		# path to datasets
		if args.data == 'baby':
			predir = './Datasets/baby/'
		elif args.data == 'sports':
			predir = './Datasets/sports/'
		elif args.data == 'tiktok':
			predir = './Datasets/tiktok/'
		else:
			raise ValueError(f"Unknown dataset: {args.data}")

		#* all datasets' file names are the same
		self.predir = predir
		#? train and test
		self.trnfile = predir + 'trnMat_new.pkl'
		self.tstfile = predir + 'tstMat_new.pkl'

		self.imagefile = predir + 'image_feat.npy'
		self.textfile = predir + 'text_feat.npy'

		if args.data == 'tiktok':  # only tiktok has audio features
			self.audiofile = predir + 'audio_feat.npy'
		
		#* delayed initialization
		self.trainMat: Optional[coo_matrix] = None
		self.torchBiAdj: Optional[torch.Tensor] = None

	def loadOneFile(self, filename):
		"""
		Load pickle file and convert it to a sparse matrix.
		"""
		with open(filename, 'rb') as fs:
			ret = (pickle.load(fs) != 0).astype(np.float32)

		if not isinstance(ret, coo_matrix): # for multi-modal features (.npy)
			ret = coo_matrix(ret)
		return ret

	def normalizeAdj(self, mat: coo_matrix): 
		"""
		Normalize a sparse adjacency matrix using the symmetric normalization method D^(-1/2) * A * D^(-1/2).

		Args:
			mat (scipy.sparse.coo_matrix): (node_num, node_num)
		"""
		csr_mat = mat.tocsr()  #* for faster computation
		# matrix's element has been set to 1.0, so degree is the number of non-zero elements in each row
		degree = np.asarray(csr_mat.sum(axis=1)).squeeze()
		dInvSqrt = np.where(degree > 0, degree**(-0.5), 0)
		dInvSqrtMat = sp.diags(dInvSqrt, offsets=0, format='csr')
		normalized_mat: csr_matrix = dInvSqrtMat @ mat @ dInvSqrtMat
		return normalized_mat.tocoo()

	def makeTorchAdj(self, mat: coo_matrix):
		"""
		Construct a sparse bipartite adjacency matrix and convert to torch sparse tensor.

		Args:
			mat (scipy.sparse.coo_matrix): (user_num, item_num)

		Returns:
			out (torch.sparse_coo_tensor): (node_num, node_num)
		"""
		#* build a sparse bipartite adjacency matrix and normalize it
		a = csr_matrix((args.user, args.user))
		b = csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])]).tocoo() # (node_num, node_num), node_num = user_num + item_num
		mat = (mat != 0) * 1.0 # convert to binary matrix (data is float) #? why do it? u-i interactions are scores?
		mat = (mat + sp.eye(mat.shape[0])) * 1.0  # set diagonal to 1 (self connection)
		mat = self.normalizeAdj(mat)

		#* make cuda tensor
		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)
		#todo: specify the tensor device
		return torch.sparse_coo_tensor(idxs, vals, shape).cuda()  # torch.Size([node_num, node_num])

	def loadFeatures(self, filename):
		feats = np.load(filename)
		return torch.tensor(feats).float().cuda(), np.shape(feats)[1]

	def LoadData(self):
		"""
		Load training and testing data, and features.
		"""
		trainMat = self.loadOneFile(self.trnfile)
		testMat = self.loadOneFile(self.tstfile)
		self.trainMat = trainMat
		args.user, args.item = trainMat.shape  # (user_num, item_num)
		self.torchBiAdj = self.makeTorchAdj(trainMat) # (node_num, node_num)

		self.trainData = TrainData(trainMat)
		self.trainLoader = dataloader.DataLoader(self.trainData, batch_size=args.batch, shuffle=True, num_workers=0)
		self.testData = TestData(testMat, trainMat)
		self.testLoader = dataloader.DataLoader(self.testData, batch_size=args.tstBat, shuffle=False, num_workers=0)

		self.image_feats, args.image_feat_dim = self.loadFeatures(self.imagefile)
		self.text_feats, args.text_feat_dim = self.loadFeatures(self.textfile)
		if args.data == 'tiktok':
			self.audio_feats, args.audio_feat_dim = self.loadFeatures(self.audiofile)

		self.diffusionData = DiffusionData(torch.FloatTensor(self.trainMat.A))
		self.diffusionLoader = dataloader.DataLoader(self.diffusionData, batch_size=args.batch, shuffle=True, num_workers=0)

class TrainData(torch_dataset):
	"""Train Dataset (with negative sampling func)"""
	def __init__(self, coomat: coo_matrix):
		# coomat -> (user_num, item_num)
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok() #* dictionary of keys (row, col) and values (data)
		self.negs = np.zeros(len(self.rows)).astype(np.int32) # neg_num == len(self.rows) == interactions (for CL)

	def negSampling(self):
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				neg_index = np.random.randint(args.item)
				#* choose non-interacted items as negative samples
				#todo: use the top -k in re-constructed u-i matrix after diffusion as negative samples
				if (u, neg_index) not in self.dokmat:
					break
			self.negs[i] = neg_index

	def __len__(self):
		"""interactions num"""
		return len(self.rows)

	def __getitem__(self, idx):
		"""idx -> (user, pos_item, neg_item)"""
		return self.rows[idx], self.cols[idx], self.negs[idx]

class TestData(torch_dataset):
	def __init__(self, testMat, trainMat):
		"""
		Test Dateset

		Args:
			testMat (coo_matrix): (user_num, item_num)
			trainMat (coo_matrix): (user_num, item_num)
		"""
		self.csrmat = (trainMat.tocsr() != 0) * 1.0 # convert to binary matrix

		test_use_its: list = [None] * testMat.shape[0] # users' interactions in test set
		test_users = set()
		for i in range(len(testMat.data)):
			user_idx = testMat.row[i]
			item_idx = testMat.col[i]
			if test_use_its[user_idx] is None:
				test_use_its[user_idx] = list()
			#* coordinate correspondence
			test_use_its[user_idx].append(item_idx)
			test_users.add(user_idx)
		test_users = np.array(list(test_users))
		self.test_users = test_users
		self.test_use_its = test_use_its

	def __len__(self):
		return len(self.test_users)

	def __getitem__(self, idx):
		return self.test_users[idx], np.reshape(self.csrmat[self.test_users[idx]].toarray(), [-1])
	
class DiffusionData(torch_dataset):
	def __init__(self, data):
		self.data = data

	def __getitem__(self, index):
		item = self.data[index]
		return item, index
	
	def __len__(self):
		return len(self.data)