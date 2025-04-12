import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix
from Conf import Config
import torch
from torch.utils.data import Dataset as torch_dataset
import torch.utils.data as dataloader

class DataHandler:
	def __init__(self, config: Config):
		"""
		Initialize dataset's path based on the dataset name from `args.data`.
		"""
		self.config = config
		self.device = torch.device(f"cuda:{self.config.base.gpu}" if torch.cuda.is_available() else "cpu")
		# path to datasets
		if self.config.data.name == 'baby':
			predir = './Datasets/baby/'
		elif self.config.data.name == 'sports':
			predir = './Datasets/sports/'
		elif self.config.data.name == 'tiktok':
			predir = './Datasets/tiktok/'
		else:
			raise ValueError(f"Unknown dataset: {self.config.data.name}")

		#* all datasets' file names are the same
		self.predir = predir
		#? train and test
		self.trainfile = predir + 'trnMat.pkl'
		self.testfile = predir + 'tstMat.pkl'

		self.imagefile = predir + 'image_feat.npy'
		self.textfile = predir + 'text_feat.npy'

		if self.config.data.name == 'tiktok':  # only tiktok has audio features
			self.audiofile = predir + 'audio_feat.npy'
		
		#* other delayed initialization are in `LoadData()`

	def loadOneFile(self, filename):
		"""
		Load pickle file and convert it to a sparse matrix.
		"""
		with open(filename, 'rb') as fs:
			ret = (pickle.load(fs) != 0).astype(np.float32)

		if not isinstance(ret, coo_matrix): # for multi-modal features (.npy)
			ret = coo_matrix(ret)
		return ret

	@staticmethod
	def normalizeAdj(mat: coo_matrix): 
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

	@staticmethod
	def makeTorchAdj(mat: coo_matrix, rows: int, cols: int, device: torch.device):
		"""
		Construct a sparse bipartite adjacency matrix and convert to torch sparse tensor.

		Args:
			mat (scipy.sparse.coo_matrix): (user_num, item_num)

		Returns:
			out (torch.sparse_coo_tensor): (node_num, node_num)
		"""
		#* build a sparse bipartite adjacency matrix and normalize it
		a = csr_matrix((rows, rows))
		b = csr_matrix((cols, cols))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])]).tocoo() # (node_num, node_num), node_num = user_num + item_num
		mat = (mat != 0) * 1.0 # convert to binary matrix (data is float) #? why do it? u-i interactions are scores?
		mat = (mat + sp.eye(mat.shape[0])) * 1.0  # set diagonal to 1 (self connection)
		mat = DataHandler.normalizeAdj(mat)

		#* make cuda tensor
		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)
		return torch.sparse_coo_tensor(idxs, vals, shape, device=device)  # torch.Size([node_num, node_num])

	def loadFeatures(self, filename):
		"""
		Load multi-modal features from .npy file and convert to torch tensor.
		
		Returns:
			tuple:
				- feats (torch.Tensor): (node_num, feat_dim)
				- feat_dim (int)
		"""
		feats: np.ndarray = np.load(filename)
		return torch.tensor(feats, dtype=torch.float, device=self.device), feats.shape[1]

	def LoadData(self):
		"""
		Load training and testing data, and features.
		"""
		trainMat = self.loadOneFile(self.trainfile)
		testMat = self.loadOneFile(self.testfile)
		self.trainMat = trainMat
		#args.user, args.item = trainMat.shape  # (user_num, item_num)
		self.config.data.user_num, self.config.data.item_num = trainMat.shape  # (user_num, item_num)
		self.torchBiAdj = self.makeTorchAdj(trainMat, self.config.data.user_num, self.config.data.item_num, self.device) # (node_num, node_num)

		self.trainData = TrainData(trainMat, self.config)
		self.trainLoader: dataloader.DataLoader[TrainData] = dataloader.DataLoader(self.trainData, batch_size=self.config.train.batch, shuffle=True, num_workers=0)
		self.testData = TestData(testMat, trainMat)
		self.testLoader: dataloader.DataLoader[TestData] = dataloader.DataLoader(self.testData, batch_size=self.config.train.test_batch, shuffle=False, num_workers=0)

		self.image_feats, self.config.data.image_feat_dim = self.loadFeatures(self.imagefile)
		self.text_feats, self.config.data.text_feat_dim = self.loadFeatures(self.textfile)
		if self.config.data.name == 'tiktok':
			self.audio_feats, self.config.data.audio_feat_dim = self.loadFeatures(self.audiofile)

		self.diffusionData = DiffusionData(torch.tensor(self.trainMat.A, dtype=torch.float, device=self.device), self.config) # .A == .toarray()
		self.diffusionLoader: dataloader.DataLoader[DiffusionData] = dataloader.DataLoader(self.diffusionData, batch_size=self.config.train.batch, shuffle=True, num_workers=0)

class TrainData(torch_dataset):
	"""Train Dataset (with negative sampling func)"""
	def __init__(self, coomat: coo_matrix, config: Config):
		self.config = config
		# coomat -> (user_num, item_num)
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok() #* dictionary of keys (row, col) and values (data)
		self.negs = np.zeros(len(self.rows)).astype(np.int32) # neg_num == len(self.rows) == interactions (for CL)

	def negSampling(self):
		"""select negative samples for each interaction"""
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				neg_index = np.random.randint(self.config.data.item_num)
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
	def __init__(self, testMat: coo_matrix, trainMat: coo_matrix):
		"""
		Test Dateset

		Args:
			testMat (coo_matrix): (user_num, item_num)
			trainMat (coo_matrix): (user_num, item_num)
		"""
		self.trainMat_csr: csr_matrix = (trainMat.tocsr() != 0) * 1.0 # convert to binary matrix

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
		self.test_user_its = test_use_its

	def __len__(self):
		return len(self.test_users)

	def __getitem__(self, idx):
		"""get user's its in train set and flatten it"""
		return self.test_users[idx], np.reshape(self.trainMat_csr[self.test_users[idx]].toarray(), [-1])
	
class DiffusionData(torch_dataset):
	"""convert trainMat to torch tensor"""
	def __init__(self, data: torch.Tensor, config: Config):
		self.data = data  # (user_num, item_num)
		self.device = torch.device(f"cuda:{config.base.gpu}" if torch.cuda.is_available() else "cpu")

	def __getitem__(self, index):
		"""
		Returns:
			tuple:
				- item (torch.Tensor)
				- index (torch.Tensor)
		"""
		item = self.data[index]
		return item, torch.tensor(index, dtype=torch.long, device=self.device)
	
	def __len__(self):
		return len(self.data)