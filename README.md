# np_gpt
This is a Numpy implementation of GPT. Inspired by Andrei Karpathy's work on implemention GPT2 with autograd using pure Python without any additional dependency, I have utilized Numpy's power on matrix computation to speed up the training. This implementation also comes with autograd for back-propagation during the training process. 

## To run unit tests

```
pytest -s -vv tests/
```

## To train a MicroGPT model on names dataset

* downloads the data at `https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt` to the location `data/names.txt`
* run the training and inference codes:
```
python src/gpt.py
```

The training loss should be improved over time. Here is a setup with 4 layers, hidden size = 16.

![Alt text](https://github.com/datduonguva/np_gpt/blob/develop/losses.png)

Once you have trained for at least 10,000 steps, you can do inference by typing seed letters in the terminal.
