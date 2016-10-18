import math
from chainer import cuda
from chainer import initializers
from chainer import link
from chainer import function
from chainer.utils import array
from chainer.utils import type_check

def _as_mat(x):
	if x.ndim == 2:
		return x
	return x.reshape(len(x), -1)

def norm_gpu(x):
	x = array.as_mat(x)
	l2norm_kernel = cuda.cupy.ReductionKernel(
		"T x, float32 eps",
		"T y",
		"x * x",
		"a + b",
		"y = sqrt(a) + eps",
		"0",
		"l2norm"
	)
	norm = cuda.cupy.broadcast_to(
		l2norm_kernel(x, 1e-5, axis=1).reshape(-1, 1),
		x.shape
	)
	return norm

cuda.cupy.linalg.norm = norm_gpu

class LinearFunction(function.Function):

	def forward(self, inputs):
		x = _as_mat(inputs[0])
		V = inputs[1]
		g = inputs[2]
		xp = cuda.get_array_module(V)
		self.normV = xp.linalg.norm(V)
		self.normalizedV = V / self.normV
		W = g * self.normalizedV
		y = x.dot(W.T).astype(x.dtype, copy=False)
		if len(inputs) == 4:
			b = inputs[3]
			y += b
		return y,

	def backward(self, inputs, grad_outputs):
		x = _as_mat(inputs[0])
		W = inputs[1]
		g = inputs[2]
		gy = grad_outputs[0]

		gx = gy.dot(W).astype(x.dtype, copy=False).reshape(inputs[0].shape)
		gW = gy.T.dot(x).astype(W.dtype, copy=False)
		gg = gW * self.normalizedV
		gV = g * gW / self.normV - g * gg * self.normalizedV / self.normV

		if len(inputs) == 3:
			gb = gy.sum(0)
			return gx, gV, gg, gb
		else:
			return gx, gV, gg

def linear(x, V, g, b=None):
	if b is None:
		return LinearFunction()(x, V, g)
	else:
		return LinearFunction()(x, V, g, b)

class Linear(link.Link):

	def __init__(self, in_size, out_size, wscale=1, bias=0, nobias=False, initialV=None, initial_bias=None):
		super(Linear, self).__init__()

		self.initialV = initialV
		self.wscale = wscale

		self.out_size = out_size
		self._V_initializer = initializers._get_initializer(initialV, math.sqrt(wscale))

		if in_size is None:
			self.add_uninitialized_param("V")
		else:
			self._initialize_params(in_size)

		if nobias:
			self.b = None
		else:
			if initial_bias is None:
				initial_bias = bias
			bias_initializer = initializers._get_initializer(initial_bias)
			self.add_param("b", out_size, initializer=bias_initializer)

		self.add_param("g", 1, initializer=initializers._get_initializer(1))

	def _initialize_params(self, in_size):
		self.add_param("V", (self.out_size, in_size), initializer=self._V_initializer)

	def __call__(self, x):
		if self.has_uninitialized_params:
			with cuda.get_device(self._device_id):
				self._initialize_params(x.size // len(x.data))
		return linear(x, self.V, self.g, self.b)