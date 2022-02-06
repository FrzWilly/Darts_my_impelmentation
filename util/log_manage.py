from datetime import datetime, timedelta
import os
import json

Model_Storage_Root = "/work/gcwscs04/models/"


def gen_log_folder_name():
    now = datetime.now()

    while os.path.exists(os.path.join(Model_Storage_Root, now.strftime("%m%d_%H%M"))):
        now += timedelta(minutes=1)

    return os.path.join(Model_Storage_Root, now.strftime("%m%d_%H%M"))


def dump_args(args, log_dir):
    args_dict = args.__dict__

    with open(os.path.join(log_dir, 'args.json'), 'w') as fp:
        json.dump(args_dict, fp)


def load_args(args, log_dir):
    with open(os.path.join(log_dir, 'args.json'), 'r') as fp:
        args_dict = json.load(fp)

    args.__dict__ = args_dict

    return args


if __name__ == '__main__':
    print(gen_log_folder_name())
