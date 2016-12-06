---
title: "Installing & Running iTorch Tutorials on Amazon AWS"
author: "Sameh Awaida"
date: "12/4/2016"
output:
  html_document: default
  pdf_document: default
---

### Helpful Links
- [Torch](http://torch.ch)
- [Torch Tutorials](http://torch.ch/docs/tutorials.html)
- [IPython](http://ipython.org/install.html)
- [iTorch](https://github.com/facebook/iTorch)
- [Running a notebook server](http://jupyter-notebook.readthedocs.io/en/latest/public_server.html)
- [Deep Learning with Torch: the 60-minute blitz](https://github.com/soumith/cvpr2015/blob/master/Deep%20Learning%20with%20Torch.ipynb)
- [Torch Video Tutorials](http://torch.ch/docs/tutorials.html)
- [Training an Object Classifier in Torch-7 on multiple GPUs over ImageNet](https://github.com/soumith/imagenet-multiGPU.torch)
- [Tutorials on Deep-Learning: from Supervised to Unsupervised Learning](https://github.com/clementfarabet/ipam-tutorials/tree/master/th_tutorials)

## 1- Install Jupyter & iTorch
```bash
cd ~/libraries
luarocks install lzmq
sudo apt-get install python-pip3
sudo pip3 install --upgrade pip
sudo pip3 install jupyter
git clone https://github.com/facebook/iTorch.git
cd iTorch
luarocks make 
```

## 2- Create Passowrd using iPython (I used 'pass' as password)
```bash
ipython
In [1]: from notebook.auth import passwd
In [2]: passwd()
Enter password:
Verify password:
Out[2]: 'sha1:1e6566bbe73f:2496114aae0b8ee5f4f0663134f19b3998101788'
```

## 3- Create config file and edit
```bash
cd ~
mkdir itorch_tutorial
cd itorch_tutorial
jupyter notebook --generate-config
vi /home/ubuntu/.jupyter/jupyter_notebook_config.py
#### Edit the config file as following:
# Set options for ip, password, and toggle off browser auto-opening
# Set ip to '*' to bind on all interfaces (ips) for the public server
c.NotebookApp.ip = '*'
c.NotebookApp.password = u'sha1:1e6566bbe73f:2496114aae0b8ee5f4f0663134f19b3998101788'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888
####
```