"""
Implementation of DNN using numpy only with autograd
TODO: write more test to make sure the gradients are correct
"""
import numpy as np
from src.gpt import *

class Model1:

    def __init__(self):

        self.linear1 = Linear(5, 10)
        self.linear2 = Linear(10, 12)
        self.linear3 = Linear(12, 4)

    def __call__(self, input_):

        y1 = self.linear1(input_)
        y2 = self.linear2(y1)
        y3 = self.linear3(y2)

        return y3

def test_1():
    """
    Create an input, run through models
    """
    input_ = Matrix(data=np.random.normal(0, 1, (3, 5)))

    my_model = Model1()
    output_ = my_model(input_)


    output_.backward()

    # create gradient wrst linear2.w
    EPS = 1e-5
    for layer in [my_model.linear1, my_model.linear2, my_model.linear3]:
        grad_method_1 = layer.w.grad

        # method 2:
        grad_method_2 = np.zeros(layer.w.data.shape)

        original_w = layer.w.data.copy()
        for i in range(grad_method_2.shape[0]):
            for j in range(grad_method_2.shape[1]):

                layer.w.data = original_w.copy()
                layer.w.data[i][j] -= EPS

                y1 = my_model(input_)

                
                layer.w.data = original_w.copy()
                layer.w.data[i][j] += EPS

                y2 = my_model(input_)

                grad_method_2[i][j] = (y2 - y1).data.sum() / (2*EPS)

        assert np.abs(grad_method_1[0, :5] - grad_method_2[0, :5]).max() < 1e-5
