# a python class that wrapps:
# train dataset gen
import numpy as np


class DetectingRegionInfo:
    def __init__(self, transmittor_position, receiver_position_1, receiver_position_2, receiver_position_3):
        self.transmittor_position = transmittor_position
        self.receiver_position_1 = receiver_position_1
        self.receiver_position_2 = receiver_position_2
        self.receiver_position_3 = receiver_position_3

        # just an alias for the above
        self.v1 = transmittor_position
        self.v2 = receiver_position_1
        self.v3 = receiver_position_2
        self.v4 = receiver_position_3
