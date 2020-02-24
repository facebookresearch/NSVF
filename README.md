# Facebook AI Differentiable Rendering (fairdr)
-----
Experimental code.

Based on the master [fairseq-py](https://github.com/pytorch/fairseq) library.
Make sure you install and run code in fairseq environment.

For installation, run
```
pip install -e .
```

To train a new model,
```
fairdr-train $DATATSET --user-dir fairdr --save-dir $MODEL_PATH 
```

To rendering a trained mode,
```
fairdr-render $DATASET --user-dir fairdr --path $MODEL_PATH/checkpoint_best.pt
```

More details, please follow the ```scripts``` folder.

<img src="http://dl.fbaipublicfiles.com/fairdr/images/rgb_512.gif" width="256" height="256">