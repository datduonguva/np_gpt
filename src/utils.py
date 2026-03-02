from google import genai
from dotenv import load_dotenv
import numpy as np
import os


def gemini_inference():
    load_dotenv()  # Loads variables from .env
    # Only run this block for Gemini Developer API
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents='Why is the sky blue?',
        config=genai.types.GenerateContentConfig(
            temperature=0,
            top_p=0.95,
            top_k=20,
        ),
    )
    print(response)


def load_mnist():
    """
    this loads the MNIST dataset for testing our NeuralNetwork implementation
    """
    from keras.datasets import mnist


    (train_features, train_targets), (test_features, test_targets) = mnist.load_data()


    train_features = train_features.reshape(60000, 784)
    print(train_features.shape)
    test_features = test_features.reshape(10000, 784)
    print(test_features.shape)


    train_features = train_features / 255.0
    test_features = test_features / 255.0

    print(train_targets.shape)
    print(test_targets.shape)

    X_train = train_features
    y_train = train_targets

    X_val = test_features
    y_val = test_targets

    np.save("x_train.npy", X_train)
    np.save("y_train.npy", y_train)

    np.save("x_val.npy", X_val)
    np.save("y_val.npy", y_val)


if __name__ == '__main__':
    load_mnist()
