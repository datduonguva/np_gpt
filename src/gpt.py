"""
Implementation of DNN using numpy only with autograd
"""
import numpy as np

class Matrix():
    """ must always have attr .data, .grad, .forward(), .backward(), children"""
    def __init__(
        self, data=None, children=None, local_grads=None
    ):
        self.data = data
        self.grad = None
        self.children = children
        self._local_grads = local_grads # grads of this node wrst children

    def __call__(self, x):
        raise ValueError("Not implemented")

    def backward(self):
        raise ValueError("Not implemented")

    def __add__(self, other):
        if not isinstance(other, Matrix):
            other = Matrix(np.zeros(self.data.shape) + other)
        return Matrix(self.data + other.data, (self, other), (1, 1))

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

class Linear():
    def __init__(self, d_in, d_out):
        self.w = Matrix(np.random.normal(0, 1, (d_in, d_out)))

    def __call__(self, other: Matrix) -> Matrix:
        return Matrix(
            other.data.dot(self.w.data),
            (self.w,  other), (other.data, self.w.data)
        )


if __name__ == '__main__':
    breakpoint()
    m_a = Matrix(data=np.random.rand(3, 5)) + 1

    linear = Linear(5, 10)

    m_b = linear(m_a)
