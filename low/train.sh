#! /bin/bash

python3 darts_model.py -NF=192 -NFL=128 -NHP=192 -M -ckpt=/work/gcwscs04/models/$1 train --resume --max_epochs=4100
