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

def _check_cudnn_acceptable_type(x_dtype, W_dtype):
	return x_dtype == W_dtype and (
		_cudnn_version >= 3000 or x_dtype != np.float16)

def _pair(x):
	if hasattr(x, "__getitem__"):
		return x
	return x, x

class Convolution2DFunction(function.Function):

	def check_type_forward(self, in_types):
		n_in = in_types.size()
		type_check.expect(2 <= n_in, n_in <= 3)

		x_type = in_types[0]
		w_type = in_types[1]
		type_check.expect(
			x_type.dtype.kind == "f",
			w_type.dtype.kind == "f",
			x_type.ndim == 4,
			w_type.ndim == 4,
			x_type.shape[1] == w_type.shape[1],
		)

		if n_in.eval() == 3:
			b_type = in_types[2]
			type_check.expect(
				b_type.dtype == x_type.dtype,
				b_type.ndim == 1,
				b_type.shape[0] == w_type.shape[0],
			)

	def forward_cpu(self, inputs):
		x, W = inputs[:2]
		b = inputs[2] if len(inputs) == 3 else None
		kh, kw = W.shape[2:]
		self.col = conv.im2col_cpu(
			x, kh, kw, self.sy, self.sx, self.ph, self.pw,
			cover_all=self.cover_all)
		y = np.tensordot(
			self.col, W, ((1, 2, 3), (1, 2, 3))).astype(x.dtype, copy=False)
		if b is not None:
			y += b
		return np.rollaxis(y, 3, 1),

	def forward_gpu(self, inputs):
		x, W = inputs[:2]
		b = inputs[2] if len(inputs) == 3 else None

		out_c, _, kh, kw = W.shape
		n, c, h, w = x.shape

		out_h = conv.get_conv_outsize(h, kh, self.sy, self.ph,
									  cover_all=self.cover_all)
		out_w = conv.get_conv_outsize(w, kw, self.sx, self.pw,
									  cover_all=self.cover_all)

		y = cuda.cupy.empty((n, out_c, out_h, out_w), dtype=x.dtype)
		if (not self.cover_all and cuda.cudnn_enabled and self.use_cudnn and
				_check_cudnn_acceptable_type(x.dtype, W.dtype)):
			x = cuda.cupy.ascontiguousarray(x)
			W = cuda.cupy.ascontiguousarray(W)
			if b is not None:
				b = cuda.cupy.ascontiguousarray(b)

			handle = cudnn.get_handle()
			x_desc = cudnn.create_tensor_descriptor(x)
			y_desc = cudnn.create_tensor_descriptor(y)

			self.filter_desc = cudnn.create_filter_descriptor(W)
			self.conv_desc = cudnn.create_convolution_descriptor(
				(self.ph, self.pw), (self.sy, self.sx), x.dtype)
			if b is not None:
				self.bias_desc = cudnn.create_tensor_descriptor(
					b[None, :, None, None])

			workspace_size = cuda.get_max_workspace_size()
			workspace = cuda.cupy.empty((workspace_size,), dtype="b")
			algo = libcudnn.getConvolutionForwardAlgorithm(
				handle, x_desc.value, self.filter_desc.value,
				self.conv_desc.value, y_desc.value, _fwd_pref,
				workspace_size)

			oz_dtype = "d" if x.dtype == "d" else "f"
			one = np.array(1, dtype=oz_dtype).ctypes
			zero = np.array(0, dtype=oz_dtype).ctypes
			libcudnn.convolutionForward(
				handle, one.data, x_desc.value, x.data.ptr,
				self.filter_desc.value, W.data.ptr, self.conv_desc.value,
				algo, workspace.data.ptr, workspace_size, zero.data,
				y_desc.value, y.data.ptr)

			# TODO(beam2d): Support unshared bias
			if b is not None:
				cudnn.add_tensor(
					handle, one.data, self.bias_desc.value, b.data.ptr,
					one.data, y_desc.value, y.data.ptr)
		else:
			# Implementation using im2col
			self.col = conv.im2col_gpu(
				x, kh, kw, self.sy, self.sx, self.ph, self.pw,
				cover_all=self.cover_all)
			W_mat = W.reshape(out_c, -1)
			col_mats = self.col.reshape(n, -1, out_h * out_w)
			y_mats = y.reshape(n, out_c, -1)
			# TODO(beam2d): Use streams or batch gemm
			for i in moves.range(n):
				y_mats[i] = W_mat.dot(col_mats[i])
			# TODO(beam2d): Support unshared bias
			if b is not None:
				y += b[:, None, None]

		return y,

	def backward_cpu(self, inputs, grad_outputs):
		x, W = inputs[:2]
		b = inputs[2] if len(inputs) == 3 else None
		gy = grad_outputs[0]
		h, w = x.shape[2:]

		gW = np.tensordot(
			gy, self.col, ((0, 2, 3), (0, 4, 5))).astype(W.dtype, copy=False)
		gcol = np.tensordot(W, gy, (0, 1)).astype(x.dtype, copy=False)
		gcol = np.rollaxis(gcol, 3)
		gx = conv.col2im_cpu(gcol, self.sy, self.sx, self.ph, self.pw, h, w)

		if b is None:
			return gx, gW
		else:
			gb = gy.sum(axis=(0, 2, 3))
			return gx, gW, gb

	def backward_gpu(self, inputs, grad_outputs):
		x, W = inputs[:2]
		b = inputs[2] if len(inputs) == 3 else None
		gy = grad_outputs[0]
		_, out_c, out_h, out_w = gy.shape
		n, c, h, w = x.shape
		kh, kw = W.shape[2:]

		gW = cuda.cupy.empty_like(W)
		if (not self.cover_all and cuda.cudnn_enabled and self.use_cudnn and
				_check_cudnn_acceptable_type(x.dtype, W.dtype)):
			x = cuda.cupy.ascontiguousarray(x)
			W = cuda.cupy.ascontiguousarray(W)
			gy = cuda.cupy.ascontiguousarray(gy)

			handle = cudnn.get_handle()
			x_desc = cudnn.create_tensor_descriptor(x)
			gy_desc = cudnn.create_tensor_descriptor(gy)
			oz_dtype = "d" if x.dtype == "d" else "f"
			one = np.array(1, dtype=oz_dtype).ctypes
			zero = np.array(0, dtype=oz_dtype).ctypes
			gx = cuda.cupy.empty_like(x)

			if _cudnn_version >= 4000:
				workspace_size = cuda.get_max_workspace_size()
				workspace = cuda.cupy.empty((workspace_size,), dtype="b")

				algo = libcudnn.getConvolutionBackwardFilterAlgorithm(
					handle, x_desc.value, gy_desc.value,
					self.conv_desc.value, self.filter_desc.value,
					_bwd_filter_pref, workspace_size)
				libcudnn.convolutionBackwardFilter_v3(
					handle, one.data, x_desc.value, x.data.ptr,
					gy_desc.value, gy.data.ptr, self.conv_desc.value,
					algo, workspace.data.ptr, workspace_size,
					zero.data, self.filter_desc.value, gW.data.ptr)

				algo = libcudnn.getConvolutionBackwardDataAlgorithm(
					handle, self.filter_desc.value, gy_desc.value,
					self.conv_desc.value, x_desc.value, _bwd_data_pref,
					workspace_size)
				libcudnn.convolutionBackwardData_v3(
					handle, one.data, self.filter_desc.value, W.data.ptr,
					gy_desc.value, gy.data.ptr, self.conv_desc.value,
					algo, workspace.data.ptr, workspace_size,
					zero.data, x_desc.value, gx.data.ptr)
			else:
				libcudnn.convolutionBackwardFilter_v2(
					handle, one.data, x_desc.value, x.data.ptr,
					gy_desc.value, gy.data.ptr, self.conv_desc.value,
					zero.data, self.filter_desc.value, gW.data.ptr)
				libcudnn.convolutionBackwardData_v2(
					handle, one.data, self.filter_desc.value, W.data.ptr,
					gy_desc.value, gy.data.ptr, self.conv_desc.value,
					zero.data, x_desc.value, gx.data.ptr)

			if b is not None:
				gb = cuda.cupy.empty_like(b)
				libcudnn.convolutionBackwardBias(
					handle, one.data, gy_desc.value, gy.data.ptr,
					zero.data, self.bias_desc.value, gb.data.ptr)
		else:
			gW_mat = gW.reshape(out_c, c * kh * kw)
			col_mats = self.col.reshape(n, c * kh * kw, out_h * out_w)
			gy_mats = gy.reshape(n, out_c, out_h * out_w)
			# TODO(beam2d): Use streams or batch gemm
			gW_mat[...] = 0
			for i in moves.range(n):
				gW_mat += cuda.cupy.dot(gy_mats[i], col_mats[i].T)

			W_mat = W.reshape(out_c, -1)
			gcol = cuda.cupy.empty_like(self.col)
			gcol_mats = gcol.reshape(n, c * kh * kw, out_h * out_w)

			for i in moves.range(n):
				gcol_mats[i] = cuda.cupy.dot(W_mat.T, gy_mats[i])

			gx = conv.col2im_gpu(
				gcol, self.sy, self.sx, self.ph, self.pw, h, w)

			if b is not None:
				gb = gy.sum(axis=(0, 2, 3))

		if b is None:
			return gx, gW
		else:
			return gx, gW, gb


def convolution_2d(x, W, b=None, stride=1, pad=0, use_cudnn=True,
				   cover_all=False):
	func = Convolution2DFunction(stride, pad, use_cudnn, cover_all)
	if b is None:
		return func(x, W)
	else:
		return func(x, W, b)

class Convolution2D(link.Link):

	def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
				 wscale=1, bias=0, nobias=False, use_cudnn=True,
				 initialW=None, initial_bias=None):
		super(Convolution2D, self).__init__()
		self.ksize = ksize
		self.stride = _pair(stride)
		self.pad = _pair(pad)
		self.use_cudnn = use_cudnn
		self.out_channels = out_channels

		# For backward compatibility
		self.initialW = initialW
		self.wscale = wscale

		# For backward compatibility, the scale of weights is proportional to
		# the square root of wscale.
		self._W_initializer = initializers._get_initializer(
			initialW, scale=math.sqrt(wscale))

		if in_channels is None:
			self.add_uninitialized_param("W")
		else:
			self._initialize_params(in_channels)

		if nobias:
			self.b = None
		else:
			if initial_bias is None:
				initial_bias = bias
			bias_initilizer = initializers._get_initializer(initial_bias)
			self.add_param("b", out_channels, initializer=bias_initilizer)

	def _initialize_params(self, in_channels):
		kh, kw = _pair(self.ksize)
		W_shape = (self.out_channels, in_channels, kh, kw)
		self.add_param("W", W_shape, initializer=self._W_initializer)

	def __call__(self, x):
		if self.has_uninitialized_params:
			with cuda.get_device(self._device_id):
				self._initialize_params(x.shape[1])
		return convolution_2d.convolution_2d(
			x, self.W, self.b, self.stride, self.pad, self.use_cudnn)


def _pair(x):
	if hasattr(x, "__getitem__"):
		return x
	return x, x