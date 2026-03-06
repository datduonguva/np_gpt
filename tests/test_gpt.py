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

        assert np.abs(grad_method_1 - grad_method_2).max() < 1e-4


def test_2():
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

def test_mnist():
    """
    This test trains an DNN model on MNIST to confirm that the loss converges.
    Here, I am using 4 Linear layer with ReLU activation
    """
    class MyModel():
        def __init__(self):
            self.linear_1 = Linear(784, 512)
            self.linear_2 = Linear(512, 256)
            self.linear_3 = Linear(256, 128)
            self.linear_4 = Linear(128, 10)
            self.relu = Relu()
            
        def __call__(self, input_: Matrix):
            x1 = self.relu(self.linear_1(input_))
            x2 = self.relu(self.linear_2(x1))
            x3 = self.relu(self.linear_3(x2))
            y = self.linear_4(x3)
            return y


    # load training data in "data/" folder. Should be easy to find on Keras
    x_train = np.load("data/x_train.npy")
    y_train = np.load("data/y_train.npy")
    n_train = x_train.shape[0]

    # Create the model, define loss function
    my_model = MyModel()
    loss_function = CategoricalEntropy()

    # Define training loop with SGD
    loss_history = []
    acc_history = []
    for step in range(2000):
        # build mini batch
        mask = np.random.randint(0, n_train, 32) 
        x_batch = x_train[mask]
        y_true = Matrix(data=np.eye(10)[y_train[mask]])

        # get model prediction
        x = Matrix(data=x_batch)
        y_pred = my_model(x) 

        # compute loss
        loss = loss_function(y_true, y_pred)
        if step % 20 == 0:
            print("loss: ", loss.data)
            loss_history.append(loss.data)
            acc_history.append((np.argmax(y_true.data, axis=-1) == np.argmax(y_pred.data, axis=-1)).mean())
            print(acc_history[-1])

        # update the weights
        loss.backward()

        for layer in [
            my_model.linear_1,
            my_model.linear_2,
            my_model.linear_3,
            my_model.linear_4
        ]:
            assert layer.w.data.shape == layer.w.grad.shape 
            layer.w.data -= 1e-6 * np.clip(layer.w.grad, -100, 100)

    assert np.mean(loss_history[-10:]) < 0.5
    assert loss_history[0] > loss_history[-1]

