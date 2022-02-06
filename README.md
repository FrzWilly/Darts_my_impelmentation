# HPCoder-PyTorch

## Project Installation
### Required packages
* Please setup a working environment via `requirements.txt`.
   * Usage: `pip install -r requirements.txt`
   * Please leave issues if there is any missing package in `requirements.txt`. 
   
### Install `torchac`
The implementation of `torchac` is coming from [L3C-PyTorch](https://github.com/fab-jul/L3C-PyTorch). Please refer to the original project for more details.

The following installation steps are quoted from [L3C-PyTorch: `torchac`](https://github.com/fab-jul/L3C-PyTorch?#the-torchac-module-fast-entropy-coding-in-pytorch):
 * *Step 1*:  Make sure a recent `gcc` is available in `$PATH` by running `$ gcc --version` (tested with version 5.5). If you want CUDA support, make sure `$ nvcc -V` gives the desired version (tested with `nvcc` version 9.0).
 * *Step 2*: 
     ```
     $ cd torchac 
     $ COMPILE_CUDA=auto python setup.py install
     ```
          
      * `COMPILE_CUDA=auto`: Use CUDA if a `gcc` between 5 and 6, and `nvcc` 9 is avaiable
      * `COMPILE_CUDA=force`: Use CUDA, don't check `gcc` or `nvcc`
      * `COMPILE_CUDA`=no: Don't use CUDA
      
        This installs a package called `torchac-backend-cpu` or `torchac-backend-gpu` in your `pip`. Both can be installed simultaneously. See also next subsection.
 * *Step 3*: Verify the installation
     ```
     $ python3 -c "import torchac"
     ```

## How to run?
### Checklist before running
* Modify the checkpoint path in `util/log_manage.py`.
* Make sure `model.py` can find the dataset correctly.

### Training
* To train the HP-Coder, run the command below under `HPCoder-Pytorch`:
  ```
  $ python3 model.py train
  ```

* To change the target bit-per-pixel, please specify the lambda at beginning:
  ```
  $ python3 model.py train --lambda=<specified lambda value>
  ```
