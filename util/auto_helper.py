import os


def get_gpu_id():
    cwd = os.getcwd()
    cwd = os.path.basename(cwd)

    assert cwd[-4:] == '_GPU'

    gpu_id = cwd[-5]
    machine_id = cwd[-6]

    print('Using GPU #{} on Machine #{}....'.format(gpu_id, machine_id))

    return int(gpu_id), int(machine_id)
