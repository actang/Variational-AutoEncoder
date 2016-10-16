# Variational Auto-Encoder

This is the course project for CS 294-129, Designing, Visualizing and Understanding Deep Neural Networks.


## Installation
### Anaconda Virtual Environment
A good way to use it is to start with an Anaconda distribution and create a virtual environment.

```
conda create -n tensorflow python=3.4
```

Use the following command to start the virtual environment.

```
source activate tensorflow
```

To exit the virtual environment, the command is the following.

```
source deactivate
```

### Packages
We need some libraries installed to run the program. You should make sure you have the followings installed. 

* Tensorflow

```conda install -c conda-forge tensorflow```

* NumPy

```conda install -c conda-forge numpy```

* SciPy

```conda install -c conda-forge scipy```

* Matloplib

```conda install -c conda-forge matloplib```

* Functional

```conda install -c conda-forge auto/functional```

Or

```pip install functional```

### Python Path
In testing phase, you may need to add the VAE source path to the system Python path. One way to do so is to modify the command shown below and type it into the terminal:

```
export PYTHONPATH="...[File Path Here].../cs294-vae/src:$PYTHONPATH"
```


## Package Architecture
Objects

* `VariationalAutoEncoder` in `vae.py` 
* `AutoEncoder` in `nn.py`
* `FullyConnectedLayer` in `layers.py`

Utility Functions

* `sampleGaussian`, `GaussianKLDivergence` in `distribution.py`

## Reference
[*Under the Hood of the Variational Autoencoder (in Prose and Code)*](http://blog.fastforwardlabs.com/post/149329060653/under-the-hood-of-the-variational-autoencoder-in).
 