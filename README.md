# Variational Auto-Encoder

## Installation
We provide two ways to set up the packages. You can either choose to 
install an Anaconda Python distribution locally and install Tensorflow 
library. Or you can directly use a Docker Image that contains Python 2.7 
and Tensorflow.

### Docker Environment
If you are using a CPU, you shoule use `gcr.io/tensorflow/tensorflow` 
Docker image. The following command will help you start running the 
container.
```
docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow bash
```

If you are using a GPU which supports NVidia drivers (ideally latest) 
and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker). Run using

```
nvidia-docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow:latest-gpu bash
```

### Anaconda Virtual Environment
A good way to start with an Anaconda distribution is to create a virtual
 environment.

```
conda create -n tensorflow python=2.7
```

Use the following command to start the virtual environment.

```
source activate tensorflow
```

To exit the virtual environment, the command is the following.

```
source deactivate
```

You can also start from any Python 2.7 distribution but you need to 
install the following libraries in order to run the program. 

* Tensorflow

```conda install -c conda-forge tensorflow```

* NumPy

```conda install -c conda-forge numpy```

* SciPy

```conda install -c conda-forge scipy```

* Matloplib

```conda install -c conda-forge matloplib```


### Python Path
In the testing phase, you may need to add the VAE source path to the 
system Python path. One way to do so is to modify the command shown 
below and type it into the terminal:

```
export PYTHONPATH="...[Path Here].../vae/src:$PYTHONPATH"
```

## Package Architecture
Objects

* `VariationalAutoEncoder` in `variationalautoencoder.py`
* `AutoEncoder` in `autoencoder.py`
* `FullyConnectedLayer` in `layers.py`
* `Distribution` in `distribution.py`

## Reference
[*Under the Hood of the Variational Autoencoder (in Prose and Code)*](http://blog.fastforwardlabs.com/post/149329060653/under-the-hood-of-the-variational-autoencoder-in).
 