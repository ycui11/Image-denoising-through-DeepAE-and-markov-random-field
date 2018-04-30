


"""Denoising Autoencoder model and training configurations."""

class ModelConfig(object):
	"""Wrapper class for model hyperparameters."""

	def __init__(self):
		"""Sets the default model hyperparameters."""
		
		# Batch size.
		self.batch_size = 50

		# Height of input images.
		self.image_height = 28

		# Width of input images.
		self.image_width = 28

		# Dimensions of input images.
		self.n_input = self.image_height * self.image_width



class TrainingConfig(object):
	"""Wrapper class for training hyperparameters."""

	def __init__(self):
		"""Sets the default training hyperparameters."""

		# Optimizer for training the model.
		self.optimizer = "Adam"

		# Learning rate for the initial phase of training.
		self.learning_rate = 0.003

		

