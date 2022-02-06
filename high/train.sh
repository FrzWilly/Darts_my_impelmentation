#! /bin/bash

python3 model.py -NF=384 -NFL=192 -NHP=256 -M -ckpt=/work/gcwscs04/models/$1 train --resume --max_epochs=4000 --lambda=8e-2
