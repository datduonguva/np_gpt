"""
Implementation of DNN using numpy only with autograd
TODO: write more test to make sure the gradients are correct
"""
import numpy as np
import matplotlib.pyplot as plt

class Matrix():
    """ must always have attr .data, .grad, .forward(), .backward(), children"""
    def __init__(
        self, data=None, children=None, local_grads=None, ops=None
    ):
        self.data = data
        self.grad = np.zeros(data.shape)
        self.children = () if children is None else children
        self.local_grads = () if local_grads is None else local_grads
        self.ops = ops # store the name of the operation, used for backprob

    def __add__(self, other):
        if not isinstance(other, Matrix):
            other = Matrix(np.zeros(self.data.shape) + other)
        return Matrix(self.data + other.data, (self, other), (1, 1))

    def __sub__(self, other): return self + (-other)
    def __mul__(self, other):
        if not isinstance(other, Matrix):
            other = Matrix(np.zeros(self.data.shape) + other)
        return Matrix( self.data* other.data, (self, other), (other.data, self.data ))
    def __neg__(self): return self * (-1)

    def __pow__(self, other): return Matrix(self.data ** other, (self, ), (other * self.data ** (other - 1),))
    def __radd__(self, other): return self + other
    def __rsub__(self, other): return self + (-other)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other ** (-1)
    def __rtruediv__(self, other): return other * self ** (-1)

    def log(self):
        return Matrix(
            np.log(self.data + 1e-9), (self, ), (1/self.data, )
        )
    def repeat(self, n, dim): 
        result = np.repeat(self.data, n, dim)
        return Matrix(
            result, (self, ), ops='repeat'
        )

    def relu(self):
        return Matrix(
            self.data * (self.data > 0), (self, ), ((self.data > 0) * 1.0, )
        )

    def backward(self):
        """
        Create a computational tree by walking to child nodes recursively,
        then compute the grad
        """
        topo = []
        visited = set([])
        def _build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    _build_topo(child)
                topo.append(v)
        _build_topo(self)

        self.grad = np.ones(self.data.shape)
        for v in reversed(topo):
            if v.ops == 'matmul':
                v.children[0].grad += v.local_grads[0].T.dot(v.grad)
                v.children[1].grad += v.grad.dot(v.local_grads[1].T)
            elif v.ops == 'rmsnorm':
                # dL/dX = sum dL/dY dY/dX = dL/dY 1/norm (delta_id - X_i * X_j/norm **2/Dim)
                y_data, norm = v.local_grads
                v.children[0].grad += 1/norm * (
                    v.grad - y_data * np.mean(y_data * v.grad, axis=-1, keepdims=True) 
                ) 
            elif v.ops == 'repeat':
                v.children[0].grad += np.sum(v.grad, axis=-1, keepdims=True)
            elif v.ops == 'softmax':
                # dL/dy.dot (s)
                term_1 = v.grad*v.data
                term_2 = np.sum(v.grad * v.data, axis=-1, keepdims=True) * v.data
                v.children[0].grad += term_1 - term_2
            else:
                for child, local_grad in zip(v.children, v.local_grads):
                    child.grad += v.grad * local_grad
class Sum:
    def __call__(self, x: Matrix):
        """
        Sum on the last dimension, keep dims
        """
        return Matrix(
            data=np.sum(x.data, axis=-1, keepdims=True,),
            children=(x, ),
            local_grads=(np.ones(x.data.shape),),
            ops='sum'
        )

class Linear():
    """ Simiar to tensorflow's Dense """
    def __init__(self, d_in, d_out):
        self.w = Matrix(np.random.normal(0, np.sqrt(2.0 / d_in), (d_in, d_out)))
    def __call__(self, other: Matrix) -> Matrix:
        return Matrix(
            other.data.dot(self.w.data),
            (self.w,  other), (other.data, self.w.data),
            ops="matmul"
        )


class Relu():
    """ Relu activate"""
    def __call__(self, x: Matrix):
        return Matrix(x.data * (x.data > 0), (x, ), ((x.data > 0) * 1.0, ))

class Softmax():
    # TODO: test this softmax
    """
    y (B, D) = softmax(x) (B, d)

    dL/dY has has (B, d)

    dL/dx_i = sum_j (dL/dy_j dy_j/d x_i)
    dL/dx_i = sum_j dL/dy_j Sj(delta - S_i)
                = v.grad * s - v.grad*s_i.sum,
    """
    def __call__(self, x: Matrix) -> Matrix:
        max_val = np.max(x.data, axis=-1, keepdims=True)
        data = x.data- max_val
        data = np.exp(data)
        data = data / np.sum(data, axis=-1, keepdims=True)
        return Matrix(
            data=data,
            children=(x, ),
            ops='softmax'
        )

   


class Dropout: 
    """ Dropout layer """
    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, x: Matrix, training=False):
        if training:
            # mask of dropped out element
            mask = (np.random.rand(x.shape) < self.ratio)*1.0
            data = x.data.copy()
            data[mask] = 0
            return Matrix(
                data, (x, ), (1.0 - mask, )
            )
        else:
            return x


class CategoricalEntropy():
    def __init__(self):
        self.sum = Sum()
    def __call__(self, y_true, y_pred, training=True):
        """
        Assuming that y_pred is already normalized by softmax and y_true is
        1-hot encoded
        """

        batch = y_true.data.shape[0]
        result = - self.sum(y_true * y_pred.log()) / batch
        return result

class RMSNorm2():
     def __call__(self, x: Matrix) -> Matrix:
        batch, dim = x.data.shape
        epsilon = 1e-7
        norm = ((Sum()(x**2) )/ dim + epsilon) ** 0.5 # (b, 1)
        norm = norm.repeat(dim, -1)  # (b, dim)
        result = x / norm # (b/dim)

        return result
   
if __name__ == '__main__':
    # this is just a test
    pass
