import math
import numpy as np
from six import moves
from chainer import cuda, Variable
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
	return cuda.cupy.sqrt(cuda.cupy.sum(x ** 2))

cuda.cupy.linalg.norm = norm_gpu

def _check_cudnn_acceptable_type(x_dtype, W_dtype):
	return x_dtype == W_dtype and (
		_cudnn_version >= 3000 or x_dtype != np.float16)

def _pair(x):
	if hasattr(x, "__getitem__"):
		return x
	return x, x

class Convolution2DFunction(convolution_2d.Convolution2DFunction):

	def check_type_forward(self, in_types):
		n_in = in_types.size()
		type_check.expect(3 <= n_in, n_in <= 4)
		x_type = in_types[0]
		v_type = in_types[1]
		g_type = in_types[2]
		type_check.expect(
			x_type.dtype.kind == "f",
			v_type.dtype.kind == "f",
			g_type.dtype.kind == "f",
			x_type.ndim == 4,
			v_type.ndim == 4,
			g_type.ndim == 1,
			x_type.shape[1] == v_type.shape[1],
		)

		if n_in.eval() == 4:
			b_type = in_types[3]
			type_check.expect(
				b_type.dtype == x_type.dtype,
				b_type.ndim == 1,
				b_type.shape[0] == v_type.shape[0],
			)

	def forward_cpu(self, inputs):
		x, V, g = inputs[:3]
		b = inputs[3] if len(inputs) == 4 else None
		
		self.normV = np.linalg.norm(V)
		self.normalizedV = V / self.normV
		self.W = g * self.normalizedV

		if b is None:
			return super(Convolution2DFunction, self).forward_cpu((x, self.W))
		return super(Convolution2DFunction, self).forward_cpu((x, self.W, b))

	def forward_gpu(self, inputs):
		x, V, g = inputs[:3]
		b = inputs[3] if len(inputs) == 4 else None

		self.normV = norm_gpu(V)
		self.normalizedV = V / self.normV
		self.W = g * self.normalizedV
		
		if b is None:
			return super(Convolution2DFunction, self).forward_gpu((x, self.W))
		return super(Convolution2DFunction, self).forward_gpu((x, self.W, b))

	def backward_cpu(self, inputs, grad_outputs):
		x, V, g = inputs[:3]
		b = inputs[3] if len(inputs) == 4 else None
		if b is None:
			gb = None
			gx, gW = super(Convolution2DFunction, self).backward_cpu((x, self.W), grad_outputs)
		else:
			gx, gW, gb = super(Convolution2DFunction, self).backward_cpu((x, self.W, b), grad_outputs)

		xp = cuda.get_array_module(x)
		gg = xp.sum(gW * self.normalizedV, keepdims=True).reshape((1,)).astype(g.dtype, copy=False)
		gV = g * (gW - gg * self.normalizedV) / self.normV
		gV = gV.astype(V.dtype, copy=False)

		if b is None:
			return gx, gV, gg
		else:
			return gx, gV, gg, gb

	def backward_gpu(self, inputs, grad_outputs):
		x, V, g = inputs[:3]
		b = inputs[3] if len(inputs) == 4 else None
		if b is None:
			gb = None
			gx, gW = super(Convolution2DFunction, self).backward_gpu((x, self.W), grad_outputs)
		else:
			gx, gW, gb = super(Convolution2DFunction, self).backward_gpu((x, self.W, b), grad_outputs)

		xp = cuda.get_array_module(x)
		gg = xp.sum(gW * self.normalizedV, keepdims=True).reshape((1,)).astype(g.dtype, copy=False)
		gV = g * (gW - gg * self.normalizedV) / self.normV
		gV = gV.astype(V.dtype, copy=False)

		if b is None:
			return gx, gV, gg
		else:
			return gx, gV, gg, gb

def convolution_2d(x, V, g, b=None, stride=1, pad=0, use_cudnn=True, cover_all=False):
	func = Convolution2DFunction(stride, pad, use_cudnn, cover_all)
	if b is None:
		return func(x, V, g)
	else:
		return func(x, V, g, b)

class Convolution2D(link.Link):

	def __init__(self, in_channels, out_channels, ksize, 
			stride=1, pad=0, wscale=1, bias=0, nobias=False, use_cudnn=True, initialV=None, dtype=np.float32):
		super(Convolution2D, self).__init__()
		self.ksize = ksize
		self.stride = _pair(stride)
		self.pad = _pair(pad)
		self.use_cudnn = use_cudnn
		self.out_channels = out_channels
		self.dtype = dtype

		self.weight_initialized = False
		self.initialV = initialV
		self.wscale = wscale
		self.nobias = nobias

		if in_channels is None:
			self.add_uninitialized_param("V")
		else:
			self._initialize_weight(in_channels)

		if nobias:
			self.b = None
		else:
			self.add_uninitialized_param("b")

		self.add_uninitialized_param("g")

	def _initialize_weight(self, in_channels):
		kh, kw = _pair(self.ksize)
		W_shape = (self.out_channels, in_channels, kh, kw)
		self.add_param("V", W_shape, initializer=initializers._get_initializer(self.initialV, math.sqrt(self.wscale)))
		self.weight_initialized = True

	def _initialize_params(self, t):
		xp = cuda.get_array_module(t)
		self.mean_t = float(xp.mean(t))
		self.std_t = math.sqrt(float(xp.var(t)))
		g = 1 / self.std_t
		b = -self.mean_t / self.std_t

		print "g <- {}, b <- {}".format(g, b)

		if self.nobias == False:
			self.add_param("b", self.out_channels, initializer=initializers.Constant(b, self.dtype))
		self.add_param("g", 1, initializer=initializers.Constant(g, self.dtype))

	def _get_W_data(self):
		V = self.V.data
		xp = cuda.get_array_module(V)
		norm = xp.linalg.norm(V)
		V = V / norm
		return self.g.data * V

	def __call__(self, x):
		if self.weight_initialized == False:
			with cuda.get_device(self._device_id):
				self._initialize_weight(x.shape[1])

		if hasattr(self, "b") == False or hasattr(self, "g") == False:
			xp = cuda.get_array_module(x.data)
			t = convolution_2d(x, self.V, Variable(xp.asarray([1]).astype(x.dtype)), None, self.stride, self.pad, self.use_cudnn)	# compute output with g = 1 and without bias
			self._initialize_params(t.data)
			return (t - self.mean_t) / self.std_t

		return convolution_2d(x, self.V, self.g, self.b, self.stride, self.pad, self.use_cudnn)
