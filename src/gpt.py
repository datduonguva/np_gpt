"""
Implementation of DNN using numpy only with autograd
"""
import numpy as np

class Layer():
    """ must always have attr .data, .grad, .forward(), .backward(), children"""
    def __init__(
        self, data=None, children=None, local_grads=None):
        self.data = data
        self.grad = None
        self.children = children
        self._local_grads = local_grads # grads of this node wrst children

    def __call__(self, x):
        raise ValueError("Not implemented")

    def backward(self):
        raise ValueError("Not implemented")

    def __add__(self, other):
        if not isinstance(other, Layer):
            other = Layer(np.zeros(self.data.shape) + other)
        return Layer(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        if not isinstance(other, Layer):
            other = Layer(np.zeros(self.shape) + other)
        return Layer( self.data* other.data, (self, other), (other.data, self.data ))


class Linear(Layer):
    def __init__(self, d_in, d_out, children=None, required_grad=False):
        super().__init__(required_grad=required_grad, children=children)
        self.w = np.random.normal(0, 1, (d_in, d_out))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[-1] == self.w.shape[0]
        self.data = x
        return x.dot(self.w)

    def backward(self, d_upstream): # d_upstream = (b, d_out)
        if self.required_grad:
            assert d_upstream.shape[-1] == self.w.shape[-1]
            self.grad = self.data.T.dot(d_upstream)
            return d_upstream.dot(self.w) # (b, d_out) x (d_out, d_in)
        return d_upstream

if __name__ == '__main__':
    m_a = Layer(data=np.random.rand(3, 5)) + 1

    breakpoint()
