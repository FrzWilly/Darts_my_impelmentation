#! /bin/bash

python3 model.py -NF=192 -NFL=128 -NHP=192 -M -ckpt=/work/gcwscs04/models/low val -SD=/work/dataset/Kodak/ -BD=stream_dir/ -TD=decode_dir/
