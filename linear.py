import math
from chainer import cuda
from chainer import initializers
from chainer import link
from chainer import function
from chainer.utils import type_check

def _as_mat(x):
	if x.ndim == 2:
		return x
	return x.reshape(len(x), -1)

class LinearFunction(function.Function):

	def forward(self, inputs):
		x = _as_mat(inputs[0])
		W = inputs[1]
		y = x.dot(W.T).astype(x.dtype, copy=False)
		if len(inputs) == 3:
			b = inputs[2]
			y += b
		return y,

	def backward(self, inputs, grad_outputs):
		x = _as_mat(inputs[0])
		W = inputs[1]
		gy = grad_outputs[0]

		gx = gy.dot(W).astype(x.dtype, copy=False).reshape(inputs[0].shape)
		gW = gy.T.dot(x).astype(W.dtype, copy=False)
		if len(inputs) == 3:
			gb = gy.sum(0)
			return gx, gW, gb
		else:
			return gx, gW

def linear(x, W, b=None):
	if b is None:
		return LinearFunction()(x, W)
	else:
		return LinearFunction()(x, W, b)

class Linear(link.Link):

	def __init__(self, in_size, out_size, wscale=1, bias=0, nobias=False,
				 initialV=None, initial_bias=None):
		super(Linear, self).__init__()

		self.initialV = initialV
		self.wscale = wscale
		self.initialG = 1

		self.out_size = out_size
		self._V_initializer = initializers._get_initializer(initialV, math.sqrt(wscale))

		if in_size is None:
			self.add_uninitialized_param("W")
		else:
			self._initialize_params(in_size)

		if nobias:
			self.b = None
		else:
			if initial_bias is None:
				initial_bias = bias
			bias_initializer = initializers._get_initializer(initial_bias)
			self.add_param("b", out_size, initializer=bias_initializer)

	def _initialize_params(self, in_size):
		self.add_param("W", (self.out_size, in_size),
					   initializer=self._V_initializer)

	def __call__(self, x):
		if self.has_uninitialized_params:
			with cuda.get_device(self._device_id):
				self._initialize_params(x.size // len(x.data))
		return linear.linear(x, self.W, self.b)