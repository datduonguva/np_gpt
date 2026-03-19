# np_gpt
numpy implementation of GPT

## To run unit tests

```
pytest -s -vv tests/
```

## to train a Micro model

* downloads the data at `https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt` to the location `data/names.txt`
* run the training and inference codes:
```
python srg/gpt.py
```

The training loss should be improved over time. Here is a setup with 4 layers,
hidden size = 16.
![Alt text](https://github.com/datduonguva/np_gpt/blob/develop/losses.png)

Once you have trained for at least 10,000 steps, you can do inference by typing seed letters in the terminal.
