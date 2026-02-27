"""
Implementation of DNN using numpy only with autograd
TODO: write more test to make sure the gradients are correct
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
                v.children[0].grad += v.local_grads[0].T.dot(v.grad)
                print(v.grad)
                print("chil 0 grad: ", v.children[0].grad)

                v.children[1].grad += v.grad.dot(v.local_grads[1].T)
                print("chil 1 grad: ", v.children[1].grad)
            else:
                for child, local_grad in zip(v.children, v.local_grads):
                    child.grad += v.grad * local_grad
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
    # this is just a test
    """
    y = x @ m

    x[0]: m -> m.grad = dy / dm (d_in, d_out) = (d_in, batch) (batch, d_out) = x.T.dot(v.grad)

               dy: (b, d_out) = x dm  (d_in, d_out)
    """
    x = Matrix(data=np.random.normal(0, 1, (3, 5)))

    linear = Linear(5, 10)

    y1 = linear(x) / 3
    y2 = y1 * y1

    y2.backward()
    grad_method_1 = x.grad.copy()
    print("grad_method_1: ", grad_method_1.shape, grad_method_1)

    # method 2:
    EPS = 1e-5
    grad_method_2 = np.zeros((3, 5))
    for i in range(3):
        for j in range(5):

            x_0 = Matrix(data=x.data.copy() )
            x_0.data[i][j] -= EPS
            x_1 = Matrix(data=x.data.copy()) 
            x_1.data[i][j] += EPS

            ta_1 = linear(x_0) /3 
            ta_2 = ta_1 * ta_1
            
            tb_1 = linear(x_1) /3 
            tb_2 = tb_1 * tb_1
 
            grad_method_2[i][j] = (tb_2 - ta_2).data.sum() / (2*EPS)

    print("grad_method_2: ", grad_method_2)
    
    assert np.abs(grad_method_1 - grad_method_2).max() < 1e-5
    # this does not work, we must calculate the grad at each layer,
    # then go to the top
