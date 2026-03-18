"""
Implementation of GPT2 using only Numpy
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import pickle

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
            np.log(self.data + 1e-9), (self, ), (1/(self.data), )
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
        dims = y_true.data.shape[:-1]
        total_dim = 1
        for dim in dims:
            total_dim *= dim
        result = - self.sum(y_true * y_pred.log()) / total_dim

        return result

class RMSNorm():
     def __call__(self, x: Matrix) -> Matrix:
        dim = x.data.shape[-1]
        epsilon = 1e-7
        norm = ((Sum()(x**2) )/ dim + epsilon) ** 0.5 # (b,..., 1)
        norm = norm.repeat(dim, -1)  # (b, dim)
        result = x / norm # (b/dim)

        return result
   

class GPT:
    """
    GPT2 implementation
    """
    def __init__(self, vocab_size, max_length=32):
        # Initialize the parameters, to store the knowledge of the model
        self.n_layer = 1     # depth of the transformer neural network (number of layers)
        self.n_embd = 32     # width of the network (embedding dimension)
        self.block_size = max_length # maximum context length 
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
        self.softmax = Softmax()
        print("Number of parameters: ", sum([a * b for a, b in shapes]))

        self.loss_func = CategoricalEntropy()

    def __call__(
        self, x: List[List[int]], y: List[List[int]] = None, mask= None
    ):
        """
        calls to GPT where x is list of token ID
        x: (B, L) 
        y: (B, L) 
        if y is not None, return Softmax loss. Else, return only the output
        """
        
        # does one hot encoded for token and positions
        batch,  ctx_len = x.shape
        mask = (1 - np.tril(np.ones((ctx_len, ctx_len))))
        mask = mask[np.newaxis, np.newaxis, :, :]
        mask = np.repeat(mask, batch, 0)
        mask = np.repeat(mask, self.n_head, 1)
        mask = Matrix(- mask * 1000)

        encoded_tokens = np.zeros((batch, ctx_len, self.vocab_size))
        if y is not None:
            encoded_target = np.zeros((batch, ctx_len, self.vocab_size))

        # 1 hot encoded 
        for b in range(batch):
            for l in range(ctx_len):
                encoded_tokens[b][l][x[b][l]] = 1
                if y is not None:
                    encoded_target[b][l][y[b][l]] = 1
            
        encoded_positions = np.zeros((batch, ctx_len, self.block_size))
        for b in range(batch):
            for l in range(ctx_len): 
                encoded_positions[b][l][l] = 1

        
        state_dict = self.state_dict

        encoded_tokens = Matrix(encoded_tokens)
        encoded_positions = Matrix(encoded_positions)
        if y is not None:
            encoded_target = Matrix(encoded_target)

        tok_emb = state_dict['wte'](encoded_tokens) # B, L, n_embd
        pos_emb = state_dict['wpe'](encoded_positions) # B, L, n_embd

        x = self.norm(tok_emb + pos_emb)

        for li in range(self.n_layer):
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
            ).transpose((0, 2, 1, 3)) # (B, N, L, H)
            k = k.reshape(
                (batch, ctx_len, self.n_head,  self.head_dim)
            ).transpose((0, 2, 3, 1)) # (B, N, H, L) 
            v = v.reshape(
                (batch, ctx_len, self.n_head,  self.head_dim)
            ).transpose((0, 2, 1, 3)) # (B, N, L, H)

            weights = (q.matmul(k)/ self.n_embd ** 0.5) # (B, N, L, L)

            x = self.softmax(weights + mask).matmul(v)  # (B, N, L, H)
            x = x.transpose((0, 2, 1, 3)).reshape((batch, ctx_len, -1)) # (B, L, D)
            x = state_dict[f'layer{li}.attn_wo'](x) # (B, L, D)
            x = x + x_residual
            x_residual = x
            x = self.norm(x)
            x = state_dict[f'layer{li}.mlp_fc1'](x).relu()
            x = state_dict[f'layer{li}.mlp_fc2'](x).relu()
            x = x + x_residual
            
        logits = state_dict['lm_head'](x)

        if y is not None: 
            logits = self.softmax(logits)
            loss = self.loss_func(encoded_target, logits)
            return logits, loss
        else:
            return logits

if __name__ == '__main__':
    with open("data/names.txt", "r") as f:
        docs = [line.strip() for line in f.readlines()]
        np.random.shuffle(docs)
    id2char = sorted(set(''.join(docs))) # 26
    BOS = len(id2char) # token id for a special Beginning of Sequence (BOS) token
    char2id= {ch: id_ for id_, ch in enumerate(id2char)} # unique characters in the dataset become token ids 0..n-1
    vocab_size = len(id2char) + 1 # total number of unique tokens, +1 is for BOS
    print(f"vocab size: {vocab_size}")

    gpt = GPT(vocab_size=vocab_size, max_length=32)
    batch_size = 8
    max_length = 16

    # train for 1000 steps
    losses = []
    for step in range(10000):
        mini_batch = [
            docs[np.random.randint(len(docs))] for _ in range(batch_size)
        ]
        # change to tokens
        token_ids = np.array([
            (
                [BOS] + 
                [char2id[ch] for ch in doc] + 
                [BOS] *  max_length
            )[:max_length]
            for doc in mini_batch
        ])
        target_ids = token_ids.copy()
        target_ids[:, :-1] = target_ids[:, 1:]
        # forward calls:
        logits, loss = gpt(x=token_ids, y=target_ids)

        loss.backward()
        losses.append(loss.data.mean())

        # sgd
        lr = 1e-3
        for name, layer in gpt.state_dict.items():
            layer.w.data -= lr * layer.w.grad
            layer.w.grad *= 0
        
        if step % 100 == 0:
            print(f"step: {step}, loss: {np.mean(losses[-100:])}")
            losses.append(loss.data.mean())

    smooth_losses = [
        np.mean
    ]
    plt.title("Loss vs step")
    plt.plot(
        list(range(100, len(losses))),
        [np.mean(losses[i - 100:i]) for i in range(100, len(losses))],
    )
    plt.savefig('losses.png')
    
    with open('model.pk', "wb") as f:
        pickle.dump(gpt, f)

    while True:
        name = input("name: ")
        while True:
            current_length = len(name)
            mini_batch = [name]
            # change to tokens
            token_ids = np.array([
                (
                    [BOS] + [char2id[ch] for ch in doc] + [BOS] *  max_length
                )[:max_length]
                for doc in mini_batch
            ])
            logits = gpt(x=token_ids)
            next_char_id = np.random.choice(
                list(range(gpt.vocab_size)),
                p=gpt.softmax(logits).data[0, current_length]
            )
            if next_char_id == BOS:
                break
            else:
                name = name + id2char[next_char_id]
                print("name: ", name)
