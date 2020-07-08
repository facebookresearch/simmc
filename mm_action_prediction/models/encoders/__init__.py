# Class decorator to register encoders.
from __future__ import absolute_import, division, print_function, unicode_literals


ENCODER_REGISTRY = {}


def register_encoder(encoder_name):
    """Register the class with the name.
    """

    def register_encoder_class(encoder_class):
        if encoder_name in ENCODER_REGISTRY:
            raise ValueError("Cant register {0} again!".format(encoder_name))
        ENCODER_REGISTRY[encoder_name] = encoder_class
        return encoder_class

    return register_encoder_class


from .history_agnostic import HistoryAgnosticEncoder
from .hierarchical_recurrent import HierarchicalRecurrentEncoder
from .memory_network import MemoryNetworkEncoder
from .tf_idf_encoder import TFIDFEncoder

__all__ = [
    "HistoryAgnosticEncoder",
    "HierarchicalRecurrentEncoder",
    "MemoryNetworkEncoder",
    "TFIDFEncoder"
]
