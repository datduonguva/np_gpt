"""
Implementation of DNN using numpy only with autograd
TODO: write more test to make sure the gradients are correct
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

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
        return Matrix(
            self.data* other.data, (self, other), (other.data, self.data )
        )
    def __neg__(self): return self * (-1)

    def __pow__(self, other):
        return Matrix(
            self.data ** other, (self, ), (other * self.data ** (other - 1),)
        )
    def __radd__(self, other): return self + other
    def __rsub__(self, other): return self + (-other)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other ** (-1)
    def __rtruediv__(self, other): return other * self ** (-1)

    def reshape(self, dims: Tuple[int]):
        """ Reshape everything """
        return Matrix(
            data=self.data.reshape(dims),
            children=(self,),
            ops='reshape' # not doing any chain rules
        )
    def transpose(self, dims: Tuple[int]):
        new_dims = [_[0] for _ in sorted(enumerate(dims), key=lambda x: x[1])]
        return Matrix(
            data=np.transpose(self.data, dims),

            children=(self,),
            local_grads=(new_dims,), # not technically localgrad
            ops='transpose'
        )
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

    def matmul(self, other):
        return Matrix(
            np.matmul(self.data, other.data),
            (self, other),
            (other.data, self.data),
            ops='matmul'
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
            if v.ops == 'linear':
                d_in = v.local_grads[0].shape[-1]
                d_out = v.grad.shape[-1]
                v.children[0].grad += (
                    v.local_grads[0].reshape((-1, d_in)).T.dot(
                        v.grad.reshape((-1, d_out))
                    )
                )
                v.children[1].grad += v.grad.dot(v.local_grads[1].T)
            elif v.ops == 'matmul':
                v.children[0].grad += np.matmul(
                    v.grad, np.swapaxes(v.local_grads[0].data, -2, -1)
                )
                v.children[1].grad += np.matmul(
                    np.swapaxes(v.local_grads[1].data, -2, -1),
                    v.grad
                )
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
            elif v.ops == 'reshape':
                v.children[0].grad += v.grad.reshape(v.children[0].grad.shape)
            elif v.ops == 'transpose':
                new_dims = v.local_grads[0]
                v.children[0].grad += np.transpose(v.grad, new_dims)
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
        self.w = Matrix(
            np.random.normal(0, np.sqrt(2.0 / d_in), (d_in, d_out))
        )
    def __call__(self, other: Matrix) -> Matrix:
        return Matrix(
            np.matmul(other.data, self.w.data),
            (self.w,  other), (other.data, self.w.data),
            ops="linear"
        )


class Relu():
    """ Relu layer """
    def __call__(self, x: Matrix):
        return Matrix(x.data * (x.data > 0), (x, ), ((x.data > 0) * 1.0, ))

class Softmax():
    def __call__(self, x: Matrix) -> Matrix:
        max_val = np.max(x.data, axis=-1, keepdims=True)
        data = x.data- max_val
        data = np.exp(data)
        data = data / np.sum(data, axis=-1, keepdims=True)
        return Matrix(data=data, children=(x, ), ops='softmax')

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

class RMSNorm():
     def __call__(self, x: Matrix) -> Matrix:
        batch, dim = x.data.shape
        epsilon = 1e-7
        norm = ((Sum()(x**2) )/ dim + epsilon) ** 0.5 # (b, 1)
        norm = norm.repeat(dim, -1)  # (b, dim)
        result = x / norm # (b/dim)

        return result
   

class GPT:
    """
    GPT2 implementation
    """
    def __init__(self, vocab_size):
        # Initialize the parameters, to store the knowledge of the model
        self.n_layer = 12     # depth of the transformer neural network (number of layers)
        self.n_embd = 16     # width of the network (embedding dimension)
        self.block_size = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
        self.n_head = 4      # number of attention heads
        self.vocab_size = vocab_size
        self.head_dim = self.n_embd // self.n_head # derived dimension of each head
        self.state_dict = {
            'wte': Linear(self.vocab_size, self.n_embd),
            'wpe': Linear(self.block_size, self.n_embd),
            'lm_head': Linear(self.n_embd, self.vocab_size,)
        }
        for i in range(self.n_layer):
            n_embd = self.n_embd
            self.state_dict[f'layer{i}.attn_wq'] = Linear(n_embd, n_embd)
            self.state_dict[f'layer{i}.attn_wk'] = Linear(n_embd, n_embd)
            self.state_dict[f'layer{i}.attn_wv'] = Linear(n_embd, n_embd)
            self.state_dict[f'layer{i}.attn_wo'] = Linear(n_embd, n_embd)
            self.state_dict[f'layer{i}.mlp_fc1'] = Linear(n_embd, n_embd * 4)
            self.state_dict[f'layer{i}.mlp_fc2'] = Linear(4 * n_embd, n_embd)

        shapes = [layer.w.data.shape for name, layer in self.state_dict.items()]

        self.norm = RMSNorm()
        print("Number of parameters: ", sum([a * b for a, b in shapes]))

    def __call__(self, x: List[List[int]]):
        """
        calls to GPT where x is list of token ID
        x: (B, L, D) 
        """
        
        # does one hot encoded for token and positions

        state_dict = seflt.state_dict
        batch, ctx_len = x.shape

        tok_emb = state_dict['wte'](x) # B, L, n_embd
        pos_emb = state_dict['wpe'](x) # B, L, n_embd

        x = self.norm(tok_emb + pos_emb)

        for i in range(self.n_layer):
            x_residual = x
            q = state_dict[f'layer{li}.attn_wq'](x) # (B, L, D)
            k = state_dict[f'layer{li}.attn_wk'](x)
            v = state_dict[f'layer{li}.attn_wv'](x)
            
            """
            q: (B, L, D) -> (B, L, N, H) -> (B , N, L, H)
            attention weights (q @ K.T) = (B, N, L, L)
            attention output = attention weights @ values 
                             = (B, N, L, L ) @ (B, N, L, H) = (B, N, L, H)
                             = transpose to (B, L, N, H) , reshape to (B, L, D)
            """

            q = q.reshape(
                (batch, ctx_len, self.n_head,  self.head_dim)
            ).reshape((0, 2, 1, 3)) # (B, N, L, H)
            k = k.reshape(
                (batch, ctx_len, self.n_head,  self.head_dim)
            ).reshape((0, 2, 3, 1)) # (B, N, H, L) 
            v = v.reshape(
                (batch, ctx_len, self.n_head,  self.head_dim)
            ).reshape((0, 2, 1, 3)) # (B, N, L, H)

            x = (q.matmul(k)/ head_dim ** 0.5).matmul(v)
            x = x.transpose((0, 2, 1, 3)).reshape((batch, ctx_len, -1))
            x = x + x_residual

            x_residual = x
            x = self.norm(x)
            x = state_dict[f'layer{li}.mlp_fc1'](x).relu()
            x = state_dict[f'layer{li}.mlp_fc2'](x).relu()
            x = x + x_residual
            
    logits = state_dict['lm_head'](x)
if __name__ == '__main__':
    gpt = GPT(vocab_size=26)
