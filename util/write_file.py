import os

import numpy as np


class BitStreamIO:
    def __init__(self, path):
        self.path = path
        self.strings = list()

        self.shape_list = list()
        self.shape_string = list()
        self.streams = list()

    def prepare_strings(self):
        shape_num = len(self.shape_string)
        stream_num = len(self.streams)

        self.strings.append(np.uint8((shape_num << 4) + stream_num).tobytes())

        self.strings += self.shape_string
        self.strings += self.streams

    def extract_strings(self):
        shape_num = int(self.strings[0][0]) // 16

        self.shape_string = self.strings[1:shape_num+1]
        self.streams = self.strings[shape_num+1:-1]

        self.extract_shape()

    @staticmethod
    def shape2string(shape):
        assert len(shape) == 4 and shape[0] == 1, shape
        shape = shape[1:]
        assert shape[0] < 2 ** 16, shape
        assert shape[1] < 2 ** 16, shape
        assert shape[2] < 2 ** 16, shape
        assert len(shape) == 3, shape
        return np.uint16(shape[0]).tobytes() + np.uint16(shape[1]).tobytes() + np.uint16(shape[2]).tobytes()

    @staticmethod
    def string2shape(string):
        shape = [1,
                 np.frombuffer(string[0:2], np.uint16)[0],
                 np.frombuffer(string[2:4], np.uint16)[0],
                 np.frombuffer(string[4:6], np.uint16)[0]]

        return shape

    def prepare_shape(self, shape_list):
        self.shape_list = shape_list

        for shape in shape_list:
            self.shape_string.append(self.shape2string(shape))

    def extract_shape(self):
        for shape_string in self.shape_string:
            self.shape_list.append(self.string2shape(shape_string))

    def write_file(self):
        assert not os.path.exists(self.path), self.path + " already exist"

        with open(self.path, 'wb') as f:
            for string in self.strings:
                f.write(string)
                f.write(b'\x12\x34\x56\x78')

    def read_file(self):
        assert os.path.exists(self.path), self.path + " doesn't exist"

        strings = b''

        with open(self.path, 'rb') as f:
            line = f.readline()
            while line:
                strings += line
                line = f.readline()

        self.strings = strings.split(b'\x12\x34\x56\x78')
