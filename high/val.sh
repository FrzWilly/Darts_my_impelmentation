#! /bin/bash

python3 model.py -NF=384 -NFL=192 -NHP=256 -M -ckpt=/work/gcwscs04/models/high val -SD=/work/dataset/Kodak/ -BD=stream_dir/ -TD=decode_dir/
