import math
import numpy as np
from six import moves
from chainer import cuda
from chainer import initializers
from chainer import link
from chainer import function
from chainer.utils import conv
from chainer.utils import type_check
from chainer.functions.connection import convolution_2d

if cuda.cudnn_enabled:
	cudnn = cuda.cudnn
	libcudnn = cuda.cudnn.cudnn
	_cudnn_version = libcudnn.getVersion()
	_fwd_pref = libcudnn.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
	if _cudnn_version >= 4000:
		_bwd_filter_pref = \
			libcudnn.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT
		_bwd_data_pref = \
			libcudnn.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT

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

def _check_cudnn_acceptable_type(x_dtype, W_dtype):
	return x_dtype == W_dtype and (
		_cudnn_version >= 3000 or x_dtype != np.float16)

def _pair(x):
	if hasattr(x, "__getitem__"):
		return x
	return x, x

class Convolution2DFunction(function.Function):

	def forward_cpu(self, inputs):
		x, V, g = inputs[:3]
		b = inputs[3] if len(inputs) == 4 else None
		
		self.normV = np.linalg.norm(V)
		self.normalizedV = V / self.normV
		self.W = g * self.normalizedV

		if b is None:
			return super(Convolution2DFunction, self).forward_cpu((x, self.W, g))
		return super(Convolution2DFunction, self).forward_cpu((x, self.W, g, b))

	def forward_gpu(self, inputs):
		x, V, g = inputs[:3]
		b = inputs[3] if len(inputs) == 4 else None

		self.normV = norm_gpu(V)
		self.normalizedV = V / self.normV
		self.W = g * self.normalizedV
		
		if b is None:
			return super(Convolution2DFunction, self).forward_gpu((x, self.W, g))
		return super(Convolution2DFunction, self).forward_gpu((x, self.W, g, b))

	def backward_cpu(self, inputs, grad_outputs):
		b = inputs[3] if len(inputs) == 4 else None
		if b is None:
			gb = None
			gx, gW = super(Convolution2DFunction, self).backward_cpu(x, self.W)
		else:
			gx, gW, gb, super(Convolution2DFunction, self).backward_cpu(x, self.W)

		gg = gW * self.normalizedV
		gV = g * gW / self.normV - g * gg * self.normalizedV / self.normV

		if len(inputs) == 3:
			return gx, gV, gg, gb
		else:
			return gx, gV, gg

	def backward_gpu(self, inputs, grad_outputs):
		b = inputs[3] if len(inputs) == 4 else None
		if b is None:
			gb = None
			gx, gW = super(Convolution2DFunction, self).backward_gpu(x, self.W)
		else:
			gx, gW, gb, super(Convolution2DFunction, self).backward_gpu(x, self.W)

		gg = gW * self.normalizedV
		gV = g * gW / self.normV - g * gg * self.normalizedV / self.normV

		if len(inputs) == 3:
			return gx, gV, gg, gb
		else:
			return gx, gV, gg

def convolution_2d(x, V, g, b=None, stride=1, pad=0, use_cudnn=True, cover_all=False):
	func = Convolution2DFunction(stride, pad, use_cudnn, cover_all)
	if b is None:
		return func(x, V, g)
	else:
		return func(x, V, g, b)

class Convolution2D(link.Link):

	def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, wscale=1, bias=0, nobias=False, use_cudnn=True, initialV=None, initial_bias=None):
		super(Convolution2D, self).__init__()
		self.ksize = ksize
		self.stride = _pair(stride)
		self.pad = _pair(pad)
		self.use_cudnn = use_cudnn
		self.out_channels = out_channels

		self.initialV = initialV
		self.wscale = wscale

		self._W_initializer = initializers._get_initializer(
			initialV, scale=math.sqrt(wscale))

		if in_channels is None:
			self.add_uninitialized_param("V")
		else:
			self._initialize_params(in_channels)

		if nobias:
			self.b = None
		else:
			if initial_bias is None:
				initial_bias = bias
			bias_initilizer = initializers._get_initializer(initial_bias)
			self.add_param("b", out_channels, initializer=bias_initilizer)
			
		self.add_param("g", 1, initializer=initializers._get_initializer(1))

	def _initialize_params(self, in_channels):
		kh, kw = _pair(self.ksize)
		W_shape = (self.out_channels, in_channels, kh, kw)
		self.add_param("V", W_shape, initializer=self._W_initializer)

	def __call__(self, x):
		if self.has_uninitialized_params:
			with cuda.get_device(self._device_id):
				self._initialize_params(x.shape[1])
		return convolution_2d(x, self.V, self.g, self.b, self.stride, self.pad, self.use_cudnn)
