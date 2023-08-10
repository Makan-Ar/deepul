import os
from os import path

import copy
import heapq
import pickle
import random
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.linear_model import LinearRegression

import torch
from torch import nn, optim

from typing import Dict, Union, List

class EarlyStopping:
	def __init__(
		self,
		min_delta: float,
		patience: int,
		mode: str,
		start_from_epoch: int=0,
		verbose: bool=False
	):
		""" Simple early stopping implementation. 

		Args:
			min_delta (float): minimum value change which would be considered an improvement
			patience (int): number of consecutive epochs without improvement 
				to wait before stopping training
			mode (str): 'min' for minimizing the loss or 'max' for maximizing it
			start_from_epoch (int, optional): number of epochs to wait before starting to 
				monitor the loss. Defaults to 0.
			verbose (bool, optional): print out when early stopping happens. Defaults to False.
		"""
		self.min_delta = min_delta
		self.patience = patience
		self.mode = mode
		self.start_from_epoch = start_from_epoch
		self.verbose = verbose
		assert mode == 'min' or mode =='max', '`mode` must be set to either "min" or "max"'

		self.mode_multi = -1 if self.mode == 'min' else 1

		self.epoch = 0
		self.best_loss = float('inf') if self.mode == 'min' else float('-inf')
		self.best_loss_epoch = -1

	def step(self, loss: float) -> bool:
		""" returns True if should stop, False otherwise

		Args:
			loss (float): loss at new epoch

		Returns:
			bool: whether should stop training or not
		"""
		if (loss - self.best_loss) * self.mode_multi > self.min_delta:
			self.best_loss = loss
			self.best_loss_epoch = self.epoch

		should_stop = self.epoch >= self.start_from_epoch and self.epoch > self.best_loss_epoch + self.patience

		if should_stop and self.verbose:
			print(f'Early stopping triggered at epoch {self.epoch}')

		self.epoch += 1
		return should_stop


def display_model_size(model: nn.Module) -> None:
	""" Displays the number of trainable and non-trainable parameters in the model """
	total_params = sum(param.numel() for param in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print((f'Total model params: {total_params:,} - ' 
			f'Non-trainable params: {total_params - trainable_params:,}'))


def set_seeds(seed):
	""" setting all seeds to make training repeatable """
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)


def plot_metric_comparison(
	train_metric: Union[List, np.ndarray],
	dev_metric: Union[List, np.ndarray], 
	n_skip: int=0, 
	metric_name: str='Loss', 
	mode: str='auto'
) -> None:
	""" Plots train vs test loss + best perfomance point of each loss

	Args:
		train_metric (Union[List, np.ndarray]): a sequence of train metric
		dev_metric (Union[List, np.ndarray]): a sequence of dev metric
		n_skip (int, optional): number of epochs to skip when plotting. Defaults to 0.
		metric_name (str, optional): name of the metric as the y axis label. Defaults to 'Loss'.
		mode (str, optional): 'min', 'max' or 'auto' for whether 'lower', 'higher' is better for 
			the metric. If set to auto, the fucntion sets it automatically based on the 
			train_metric.Defaults to 'auto'.
	"""
	assert mode in ['min', 'max', 'auto'], '`mode` must be set to either min, max or auto'

	def report_na(metric, name):
		n_na = np.sum(np.isnan(metric))
		if n_na > 0:
			print(f'{name} {metric_name} has {n_na} NaN values')
		
	# print info on nan values if any
	report_na(train_metric, 'Train')
	report_na(dev_metric, 'Dev')
	
	if mode == 'auto':
		non_na_idx = ~np.isnan(train_metric)	
		x = np.arange(len(train_metric))[non_na_idx]
		y = np.array(train_metric)[non_na_idx]

		reg = LinearRegression().fit(x[:, np.newaxis], y)
		mode = 'min' if reg.coef_[0] < 0 else 'max'

	train_metric, dev_metric = train_metric[n_skip:], dev_metric[n_skip:]

	argbestfunc = np.argmin if mode == 'min' else np.argmax
	train_arg_best = argbestfunc(train_metric)
	dev_arg_best = argbestfunc(dev_metric)

	plt.plot(np.arange(n_skip, len(train_metric) + n_skip), train_metric, 'g-', label='Train')
	plt.scatter(train_arg_best + n_skip, train_metric[train_arg_best], marker='d', c='orange',
				label=f'Best Train: {train_metric[train_arg_best]:.3e} @ E{train_arg_best + n_skip}')

	plt.plot(np.arange(n_skip, len(dev_metric) + n_skip), dev_metric, 'b-', label='Dev')
	plt.scatter(dev_arg_best + n_skip, dev_metric[dev_arg_best], marker='*', color='m',
			label=f'Best Dev: {dev_metric[dev_arg_best]:.3e} @ E{dev_arg_best + n_skip}')


	plt.ylabel(metric_name if metric_name else 'Loss')
	plt.xlabel('epoch')
	plt.legend()
	plt.show()

def compute_avg_metrics(
	metrics: Dict,
	batch_sizes: list,
	dataset_size: int
) -> Dict:
	""" computes avg metrics based on exact batch sizes

		Checks if the sum of all batch sizes match the dataset size, if not
		it issues a warning and simply reports the avg over the metric list

	Args:
		metrics (Dict): dict of all tracked metrics {'metric name': [metrics values]}
		batch_sizes (list): list of the exact batch sizes in the epoch
		dataset_size (int): size of the dataset

	Returns:
		Dict: avraged metrics
	"""
	# compute averages for metrics
	if np.sum(batch_sizes) != dataset_size:
		warnings.warn(('Exact batch sizes could not be deffered automatically from the '
		 			   'data loader. Metrics will be just averages over batches.'))
		for metric_name, metric_list in metrics.items():
			metrics[metric_name] = np.mean(metric_list)
	else:
		for metric_name, metric_list in metrics.items():
			batch_metric = [metric_list[i] * batch_sizes[i] for i in range(len(metric_list))]
			metrics[metric_name] = np.sum(batch_metric) / dataset_size
	
	return metrics

def evaluate(
	model: nn.Module,
	data_loader: torch.utils.data.DataLoader
) -> Dict:
	""" evaluates a model on data_loader and reports the average across batches

	Args:
		model (nn.Module): pytroch model instance
		data_loader (torch.utils.data.DataLoader): dataloader to evaluate the model
		verbose (bool, optional): prints. Defaults to False.

	Returns:
		Dict: avg metrics across batches
	"""
	model.eval()

	metrics = defaultdict(list)
	batch_sizes = []  # to compute losses based on exact batch sizes
	with torch.no_grad():
		for batch_i, batch in enumerate(data_loader):
			for k, v in batch.items():
				batch[k] = v.to(model.device)
				if len(batch_sizes) == batch_i:
					batch_sizes.append(batch[k].shape[0])
			batch_metrics = model.loss(**batch)

			for metric_name, metric_value in batch_metrics.items():
				metrics[metric_name].append(metric_value.item())

	return compute_avg_metrics(metrics, batch_sizes, len(data_loader.dataset))


class Trainer:
	def __init__(
		self,
		model: nn.Module,
		train_loader: torch.utils.data.DataLoader,
		dev_loader: torch.utils.data.DataLoader, 
		optimizer: torch.optim.Optimizer,
		test_loader: torch.utils.data.DataLoader=None, 
		lr_scheduler: torch.optim.lr_scheduler=None,
		early_stopping: EarlyStopping=None,
		grad_clip: float=None,
		store_model_path: str=None, 
		store_top_k: int=0, 
		mode: str='min',
		seed: int= None,
		verbose: bool=True
	) -> None:
		""" a trainer class which helps manage training a model, tracking relative metrics and storing top models

		Args:
			model (nn.Module): an instance of a model to train
			train_loader (torch.utils.data.DataLoader): training dataset
			dev_loader (torch.utils.data.DataLoader): development dataset
			optimizer (torch.optim.Optimizer): an instance of a gradient descent optimizer
			test_loader (torch.utils.data.DataLoader, optional): test dataset. To be used for tracking loss on 
				test dataset. Defaults to None.
			lr_scheduler (torch.optim.lr_scheduler): an instance of a gradient descent optimizer. Defaults to None.
			early_stopping (EarlyStopping, optional): an instance of EarlyStopping. Defaults to None.
			grad_clip (float, optional): max norm of the gradients to be clipped. If None, no clipping
				is performed. Defaults to None.
			store_model_path (str, optional): path to store best models. If None, nothing will be 
				stored. Defaults to None.
			store_top_k (int, optional): stores top k models based on dev loss. Defaults to 0.
			mode (str, optional): 'min' for minimizing the loss or 'max' for maximizing it. Defaults to 'min'.
			seed (int, optional): value to set all seeds before training to make it repeatable. If None, 
				no seeds will be set. Defaults to None.
			verbose (bool, optional): if True, various info will be displayed during training. Defaults to True.
		"""
		assert store_model_path is None or store_top_k >= 1, '`store_top_k` must be bigger than 0 if `store_model_path` is passed'
		assert mode == 'min' or mode =='max', '`mode` must be set to either "min" or "max"'
	
		self._model = model
		self._train_loader = train_loader
		self._dev_loader = dev_loader
		self._optimizer = optimizer
		self._test_loader = test_loader
		self._lr_scheduler = lr_scheduler
		self._early_stopping = early_stopping
		self._grad_clip = grad_clip
		self._store_model_path = store_model_path
		self._store_top_k = store_top_k
		self._mode = mode
		self._seed = seed
		self._verbose = verbose

		if self._verbose:
			display_model_size(self._model)

		if seed is not None:
			set_seeds(self._seed)
		
		if self._store_model_path is not None:
			self._checkpoint_path = path.join(self._store_model_path, 'checkpoints')
			if not path.exists(self._checkpoint_path):
				os.makedirs(self._checkpoint_path)
		else:
			self._store_top_k = 0
			self._checkpoint_path = None

		self._epoch_cnt = 0

		self._best_models_heap = []
		self._mode_multi = -1 if self._mode == 'min' else 1

		self.train_metrics = defaultdict(list)
		self.dev_metrics = defaultdict(list)
		self.test_metrics = defaultdict(list)

	def _train_single_epoch(
		self,
		current_epoch: int,
		p_bar_leave: bool=True,
		pbar_desc_add_on: str=None,
	 ) -> Dict:
		""" Runs a single training loop

		Args:
			current_epoch (int): current epoch
			p_bar_leave (bool, optional): whether to leave tqdm's progress 
				bar. Defaults to True.
			pbar_desc_add_on (str, optional): additional description to add to 
				tqdm's progress bar. Defaults to None.

		Returns:
			Dict: current epoch training metrics averaged all batches
		"""
		self._model.train()
		dset_size = len(self._train_loader.dataset)

		metrics = defaultdict(list)
		batch_sizes = []  # to compute metrics based on eaxct batch sizes

		pbar = tqdm(total=dset_size, leave=p_bar_leave, disable=not self._verbose)
		for batch_i, batch in enumerate(self._train_loader):
			for k, v in batch.items():
				batch[k] = v.to(self._model.device).contiguous()
				if len(batch_sizes) == batch_i:
					batch_sizes.append(batch[k].shape[0])

			# forward + backward pass
			batch_metrics = self._model.loss(**batch)
			self._optimizer.zero_grad()
			batch_metrics['loss'].backward()
			if self._grad_clip is not None:
				nn.utils.clip_grad_norm_(self._model.parameters(), self._grad_clip)
			self._optimizer.step()

			# track metrics + update tqdm desc
			pbar_metric_desc = []
			for metric_name, metric_value in batch_metrics.items():
				metrics[metric_name].append(metric_value.item())
				pbar_metric_desc.append(f'{metric_name}: {np.mean(metrics[metric_name][-100:]):.3e}')
			pbar_metric_desc = ' - '.join(pbar_metric_desc)
			
			pbar.update(batch_sizes[-1])
			pbar_desc = f'E{current_epoch} | Train ' + pbar_metric_desc
			if pbar_desc_add_on:
				pbar_desc += f' | {pbar_desc_add_on}'
			pbar.set_description(pbar_desc)
		pbar.close()

		self._epoch_cnt += 1

		return compute_avg_metrics(metrics, batch_sizes, dset_size)
	
	def _store(
		self, 
		dev_loss: float, 
		train_loss: float, 
		epoch: int
	) -> None:
		""" Updates the best model heap and stores the model instance if the model has a better loss
			than the worst model on the heap. The size of the heap is defined by `store_top_k`

		Args:
			dev_loss (float): current model's dev loss
			train_loss (float): current model's trian loss
			epoch (int): current epoch
		"""
		is_heap_not_full = len(self._best_models_heap) < self._store_top_k
		is_dev_loss_improved = is_heap_not_full or (dev_loss - self._best_models_heap[0][0] * self._mode_multi) * self._mode_multi > 0

		if not is_dev_loss_improved:
			return
		
		if len(self._best_models_heap) == self._store_top_k:
			_, path_to_remove = heapq.heappop(self._best_models_heap)
			
			if path.exists(path_to_remove):
				os.remove(path_to_remove)
		
		model_path = path.join(self._checkpoint_path, f'e{epoch}_dev_{dev_loss:.3e}_train_{train_loss:.3e}.ckpt')
		heapq.heappush(self._best_models_heap, (self._mode_multi * dev_loss, model_path))
		torch.save(self._model.state_dict(), model_path)

	def train(
		self,
		epochs: int
	):
		""" Trains the model for the specified number of epochs

		Args:
			epochs (int): number of epochs to train the model
		"""
		dev_pref_report = None

		for epoch in range(self._epoch_cnt, self._epoch_cnt + epochs):
			is_last_epoch = epoch == epochs - 1

			train_epoch_metrics = self._train_single_epoch(
				current_epoch=epoch, 
				p_bar_leave=is_last_epoch, 
				pbar_desc_add_on=dev_pref_report
			)

			dev_epoch_metrics = evaluate(self._model, self._dev_loader)
			
			if self._lr_scheduler is not None:
				self._lr_scheduler.step(dev_epoch_metrics['loss'])
			
			# track epoch train metrics
			for metric_name, metric_value in train_epoch_metrics.items():
				self.train_metrics[metric_name].append(metric_value)

			# track epoch dev metrics
			dev_pref_report = []
			for metric_name, metric_value in dev_epoch_metrics.items():
				self.dev_metrics[metric_name].append(metric_value)
				dev_pref_report.append(f'{metric_name}: {metric_value:.3e}')
			dev_pref_report = 'Dev ' + ' - '.join(dev_pref_report)

			# track epoch test metrics if test loader is provided
			if self._test_loader is not None:
				test_epoch_metrics = evaluate(self._model, self._test_loader)
				for metric_name, metric_value in test_epoch_metrics.items():
					self.test_metrics[metric_name].append(metric_value)

			# add info on best performing metrics
			best_arg_func = np.argmin if self._mode == 'min' else np.argmax
			arg_best_dev_loss = best_arg_func(self.dev_metrics['loss'])
			dev_pref_report += f' | Best LOSS E{arg_best_dev_loss}: {self.dev_metrics["loss"][arg_best_dev_loss]:.3e}'
			
			if self._early_stopping is not None and self._early_stopping.step(dev_epoch_metrics['loss']):
				break
		
			if self._store_top_k > 0:
				self._store(dev_epoch_metrics['loss'], train_epoch_metrics['loss'], epoch)
			
		if self._verbose:
			print(dev_pref_report)
				
		if self._store_top_k > 0:
			with open(path.join(self._store_model_path, 'train_metrics.pckl'), 'wb') as fp:
				pickle.dump(self.train_metrics, fp, protocol=pickle.HIGHEST_PROTOCOL)

			with open(path.join(self._store_model_path, 'dev_metrics.pckl'), 'wb') as fp:
				pickle.dump(self.dev_metrics, fp, protocol=pickle.HIGHEST_PROTOCOL)

			if self._test_loader is not None:
				with open(path.join(self._store_model_path, 'test_metrics.pckl'), 'wb') as fp:
					pickle.dump(self.test_metrics, fp, protocol=pickle.HIGHEST_PROTOCOL)
		return
	
	def get_best_model_path(self) -> str:
		""" returns the path to the best perfoming model across all epochs based on dev loss

		Returns:
			str: path to the best performing model instance
		"""
		assert self._store_model_path is not None, 'No model was stored. Set `store_model_path` if you would like to store models'

		stored_models = sorted(self._best_models_heap, key=lambda x: x[0])
		best_model_path = stored_models[0][1]
		return best_model_path
	
	def load_best_model(
		self, 
		device: torch.device=None,
		set_to_eval: bool=True
	) ->  nn.Module:
		""" Loads the best perfoming model across all epochs based on dev loss

		Args:
			device (torch.device, optional): device to load the model into. If None, it loads into the same 
				device as the Trainer model. Defaults to None.
			set_to_eval (bool, optional): whether to set the model to eval mode. Defaults to True.

		Returns:
			nn.Module: best performing model instance
		"""
		best_model_path = self.get_best_model_path()
		device = device if device is not None else self._model.device

		if self._verbose:
			print('Best model path:', best_model_path)
		
		best_model = copy.deepcopy(self._model)
		best_model.load_state_dict(torch.load(best_model_path))
		best_model.device = device
		best_model.to(device)
		if set_to_eval:
			best_model.eval()

		return best_model

	def plot_all_metric_comparisons(
		self, 
		n_skip: int=0
	) -> None:
		""" plots metric comparison for all tracked metrics

		Args:
			n_skip (int, optional): number of epochs to skip when plotting. Defaults to 0.
		"""
		if len(self.train_metrics['loss']) == 0:
			print('No training loss has been recorded yet on this Trainer instance.')
			return

		for metric_name in self.train_metrics:
			plot_metric_comparison(
				train_metric=self.train_metrics[metric_name],
				dev_metric=self.dev_metrics[metric_name], 
				n_skip=n_skip, 
				metric_name=metric_name, 
				mode='auto'
			)
			plt.clf()

