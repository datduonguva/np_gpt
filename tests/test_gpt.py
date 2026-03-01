"""
Implementation of DNN using numpy only with autograd
TODO: write more test to make sure the gradients are correct
"""
import numpy as np
from src.gpt import *

def test_1():
    """
    Create an input, run through models
    """
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

        assert np.abs(grad_method_1 - grad_method_2).max() < 1e-5


def test_1():
    """
    Create an input, run through models
    """
    class MyModel:

        def __init__(self):

            self.linear1 = Linear(5, 10)
            self.linear1a = Linear(5, 20)

            self.linear2 = Linear(10, 12)
            self.linear2a = Linear(20, 12)

            self.linear3 = Linear(12, 4)

        def __call__(self, input_):

            y1 = self.linear1(input_)
            y2 = self.linear2(y1)
            
            y1a = self.linear1a(input_)
            y2a = self.linear2a(y1a)


            y3 = self.linear3(y2**2 +   y2a)

            return y3

    input_ = Matrix(data=np.random.normal(0, 1, (3, 5)))

    my_model = MyModel()
    output_ = my_model(input_)


    output_.backward()

    EPS = 1e-6
    # shallower layers might produce more error compared to deeper layers
    for layer, max_error in [
        (my_model.linear1, 1e-3),
        (my_model.linear1a, 1e-3),
        (my_model.linear2, 1e-3),
        (my_model.linear2a, 1e-4),
        (my_model.linear3, 1e-4)
    ]:
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

        assert np.abs(grad_method_1 - grad_method_2).max() < max_error
