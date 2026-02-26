"""
Implementation of DNN using numpy only with autograd
"""
import numpy as np

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

    def __call__(self, x):
        raise ValueError("Not implemented")

    def backward(self):
        """
        This is the loss function
        after calculing the output, it should have the .data
        Now, we are calculating the grad of the data with respect to its 
        child

        for example:
        L = (y_true - y_pred) **2,  = 

        L.backward()

        find the grad from all childs
        """
        topo = []
        visited = set([])
        def _build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    _build_topo(child)
                topo.append(v)
        _build_topo(self) # [input-layers ->outputs]

        self.grad = np.ones(self.data.shape)
        for v in reversed(topo):
            if v.ops == 'matmul':
                # (d_in, d_out)  == (batch, d_in), (batch_dout) 
                print("v.children[0].grad.shape: ", v.children[0].grad.shape) 
                print("v.local_grads[0]: ", v.local_grads[0].shape)
                print("v.children[1].grad", v.children[1].grad.shape)
                print("v.grad", v.grad.shape)
                v.children[0].grad += v.local_grads[0].T.dot(v.grad)
                print(v.local_grads[0].T)
                print("--------")
                print("v.grad")
                print(v.grad)
                v.children[1].grad += v.grad.dot(v.local_grads[1].T)
            else:
                for child, local_grad in zip(v.children, v.local_grads):
                    child.grad += v.grad * local_grad
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
                       # (b, din)
    def __call__(self, other: Matrix) -> Matrix:
        return Matrix(
            other.data.dot(self.w.data),
            (self.w,  other), (other.data, self.w.data),
            ops="matmul"
        )


if __name__ == '__main__':
    breakpoint()
    x = Matrix(data=np.zeros((3, 5))) + 1

    linear = Linear(5, 10)

    y = linear(x)
    y.backward()

    print(y.children[0].grad.shape)

    # this does not work, we must calculate the grad at each layer,
    # then go to the top
