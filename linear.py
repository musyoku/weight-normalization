import math
from chainer import cuda, Variable
from chainer import initializers
from chainer import link
from chainer import function
from chainer.utils import array
from chainer.utils import type_check
from chainer.functions.connection import linear

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

class LinearFunction(linear.LinearFunction):

	def check_type_forward(self, in_types):
		n_in = in_types.size()
		type_check.expect(3 <= n_in, n_in <= 4)
		x_type, w_type, g_type = in_types[:3]
		
		type_check.expect(
			x_type.dtype.kind == "f",
			w_type.dtype.kind == "f",
			g_type.dtype.kind == "f",
			x_type.ndim >= 2,
			w_type.ndim == 2,
			g_type.ndim == 1,
			type_check.prod(x_type.shape[1:]) == w_type.shape[1],
		)
		if n_in.eval() == 4:
			b_type = in_types[3]
			type_check.expect(
				b_type.dtype == x_type.dtype,
				b_type.ndim == 1,
				b_type.shape[0] == w_type.shape[0],
			)

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

	def __init__(self, in_size, out_size, wscale=1, bias=0, nobias=False, initialV=None):
		super(Linear, self).__init__()

		self.params_initialized = False
		self.weight_initialized = False
		self.initialV = initialV
		self.wscale = wscale
		self.nobias = nobias

		self.out_size = out_size
		self._V_initializer = initializers._get_initializer(initialV, math.sqrt(wscale))


		if in_size is None:
			self.add_uninitialized_param("V")
		else:
			self.initialize_weight(in_size)

		if nobias:
			self.b = None
		else:
			self.add_uninitialized_param("b")

		self.add_uninitialized_param("g")

	def initialize_weight(self, in_size):
		self.add_param("V", (self.out_size, in_size), initializer=self._V_initializer)
		self.weight_initialized = True

	def initialize_params(self, t):
		xp = cuda.get_array_module(t)
		self.mean_t = float(xp.mean(t))
		self.std_t = math.sqrt(float(xp.var(t)))
		g = 1 / self.std_t
		b = -self.mean_t / self.std_t

		if self.nobias == False:
			self.add_param("b", self.out_size, initializer=initializers._get_initializer(b))
		self.add_param("g", 1, initializer=initializers._get_initializer(g))
		
		self.params_initialized = True

	def get_W_data(self):
		V = self.V.data
		xp = cuda.get_array_module(V)
		norm = xp.linalg.norm(V)
		V = V / norm
		return self.g.data * V

	def __call__(self, x):
		if self.weight_initialized == False:
			with cuda.get_device(self._device_id):
				self.initialize_weight(x.size // len(x.data))

		if self.params_initialized == False:
			xp = cuda.get_array_module(x)
			t = linear(x, self.V, Variable(xp.asarray([1]).astype(x.dtype)))	# compute output with g = 1 and without bias
			self.initialize_params(t.data)
			return (t - self.mean_t) / self.std_t

		return linear(x, self.V, self.g, self.b)