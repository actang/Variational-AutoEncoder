# Here is the guide to use EC2 GPU instance to train tensorflow neural networks
# AMI: ami-133ae673

pip install matplotlib
pip install tqdm
pip install -U pip
sudo pip install prettytensor

export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.0rc0-cp27-none-linux_x86_64.whl
sudo pip install --upgrade $TF_BINARY_URL

mkdir ~/.config
mkdir ~/.config/matplotlib
echo "backend : Agg" >> ~/.config/matplotlib/matplotlibrc