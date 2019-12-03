"""Implementation of counter-based random number generator Philox2x32.

Random123 C++ framework, containing implementation of Philox RNG:

https://www.deshawresearch.com/resources_random123.html

Reference article:

John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw. 2011.
Parallel random numbers: as easy as 1, 2, 3.
In Proceedings of 2011 International Conference for High Performance Computing,
Networking, Storage and Analysis (SC ’11). Association for Computing Machinery,
New York, NY, USA, Article 16, 1–12.
DOI:https://doi.org/10.1145/2063384.2063405

"""

import numpy as np
from tvm import relay

# Philox constants for 32-bits generation.

PHILOX_M2x32_0 = np.uint64(0xd256d193)
PHILOX_W32_0 = np.uint32(0x9E3779B9)
PHILOX2x32_DEFAULT_ROUNDS = 10

# Constants for conversion from uint32 to uniform float32.
F1 = np.float32(1)
F2 = np.float32(2)
F128 = np.float32(128)
F1024 = np.float32(1024)
R123_0x1p_31f = F1 / (F1024 * F1024 * F1024 * F2)
R123_0x1p_24f = F128 * R123_0x1p_31f

# Relay constants.
RELAY_UINT32_8 = relay.const(8, 'uint32')
RELAY_UINT64_32 = relay.const(32, 'uint64')
RELAY_UINT64_CLEAR_HIGH = relay.const(0x00000000ffffffff, 'uint64')
RELAY_PHILOX_M2x32_0 = relay.const(PHILOX_M2x32_0, 'uint64')
RELAY_PHILOX_W32_0 = relay.const(PHILOX_W32_0, 'uint32')
RELAY_R123_0x1p_24f = relay.const(R123_0x1p_24f, 'float32')


def generate_function(impl, inputs):
    """Generate relay function.

    Use impl callback with inputs as parameters to get symbolic output,
    and then generate relay function using inputs and output.
    :type impl: callable
    :type inputs: list
    """
    output = impl(*inputs)
    return relay.Function(list(inputs), output)


class Philox2x32:
    """Implementation of Philox2x32 RNG."""

    __slots__ = ('output_size', 'n', 'philox_2x_round')

    @staticmethod
    def get_counter_size(output_size):
        """Compute size of counters array to be used to generate random values.

        Compute size of a 64-bits integers vector encoding
        as many 32-bits integers as given output_size.

        If output_size is even, then vector size is n == output_size / 2,

        If output_size is odd, then vector size is (output_size + 1) / 2.
        In such case, vector is encoding output_size + 1 values.
        We will just ignore the last value in generated ones
        to get desired output_size values.
        """
        return (output_size + (output_size % 2)) // 2

    def generate_numpy_counter_array(self, counter):
        """Generate numpy uint64 counter array for Philox2x32 RNG.

        Generate a numpy vector of 64-bits integers
        which encodes couples (counter, i) for i in range(n).

        counter must be an integer.
        """
        return ((np.full((self.n,), counter, np.uint64) << 32)
                | np.arange(self.n, dtype='uint64'))

    def generate_relay_counter_array(self, counter):
        """Generate relay symbolic uint64 counter array for Philox2x32 RNG.

        Generate a relay vector of 64-bits integers
        which encodes couples (counter, i) for i in range(n)

        counter must be a relay expression
        (e.g. a relay constant or variable).
        """
        c = relay.cast(counter, 'uint64')
        b = relay.op.transform.full(c, (self.n,), 'uint64')
        d = relay.left_shift(b, RELAY_UINT64_32)
        e = relay.arange(relay.const(self.n, 'uint64'), dtype='uint64')
        return relay.bitwise_or(d, e)

    def __init__(self, output_size: int):
        """Initialize Philox2x32 RNG for given output size."""
        self.output_size = output_size
        self.n = self.get_counter_size(output_size)
        ctr_type = relay.ty.TensorType((self.n,), 'uint64')
        local_ctr = relay.var('ctr', type_annotation=ctr_type)
        local_key = relay.var('key', dtype='uint32', shape=())
        self.philox_2x_round = generate_function(
            self.__impl_philox_2x_round, [local_ctr, local_key])

    def __impl_philox_2x_round(self, ctr, key):
        """Compute a round in Philox2x32.

        :param ctr: uint64 vector
        :param key: uint32 scalar
        :return:
        """
        ctr_0 = relay.right_shift(ctr, RELAY_UINT64_32)
        ctr_1 = relay.bitwise_and(ctr, RELAY_UINT64_CLEAR_HIGH)

        # mul_hi_lo
        product = relay.multiply(RELAY_PHILOX_M2x32_0, ctr_0)

        key_64 = relay.cast(key, 'uint64')
        ctr_1_xor_key = relay.bitwise_xor(ctr_1, key_64)
        ctr_1_xor_key_up = relay.left_shift(ctr_1_xor_key, RELAY_UINT64_32)
        return relay.bitwise_xor(product, ctr_1_xor_key_up)

    def __uint64_to_2xuint32_vector(self, ctr):
        """Convert a uint64 vector to a corresponding uint32 vector.

        Given uint64 vector with size n is converted to a
        uint32 vector with size 2n.
        Each uint64 is split into couple (32 high bits, 32 low bits).
        Output values order is the same as input, ie., both values
        from a uint64 remain consecutive in output vector.
        """
        hi = relay.right_shift(ctr, RELAY_UINT64_32)
        lo = relay.bitwise_and(ctr, RELAY_UINT64_CLEAR_HIGH)
        hi_32 = relay.cast(hi, 'uint32')
        lo_32 = relay.cast(lo, 'uint32')
        vector_hi_32 = relay.reshape(hi_32, (self.n, 1))
        vector_lo_32 = relay.reshape(lo_32, (self.n, 1))
        tensor = relay.concatenate([vector_hi_32, vector_lo_32], 1)
        return relay.reshape(tensor, (2 * self.n))

    def __uint32_to_float32(self, tensor):
        """Convert uint32 to float32 in interval [0, 1).

        Apply (i >> 8) * R123_0x1p_24f to each uint32 i.
        """
        a = relay.right_shift(tensor, RELAY_UINT32_8)
        b = relay.cast(a, 'float32')
        return relay.multiply(b, RELAY_R123_0x1p_24f)

    def philox_2x_bump_key(self, key):
        """Bump key."""
        return relay.add(key, RELAY_PHILOX_W32_0)

    def impl_philox_2x_r(self, r, ctr, key):
        """Generate random values.

        :param r: number of rounds to execute
        :param ctr: counter array: uint64 vector
        :param key: key: uint32 scalar
        :return: random values in a uint64 vector with same size as ctr.
        """
        assert 0 <= r <= 16
        if r > 0:
            ctr = self.philox_2x_round(ctr, key)
        if r > 1:
            key = self.philox_2x_bump_key(key)
            ctr = self.philox_2x_round(ctr, key)
        if r > 2:
            key = self.philox_2x_bump_key(key)
            ctr = self.philox_2x_round(ctr, key)
        if r > 3:
            key = self.philox_2x_bump_key(key)
            ctr = self.philox_2x_round(ctr, key)
        if r > 4:
            key = self.philox_2x_bump_key(key)
            ctr = self.philox_2x_round(ctr, key)
        if r > 5:
            key = self.philox_2x_bump_key(key)
            ctr = self.philox_2x_round(ctr, key)
        if r > 6:
            key = self.philox_2x_bump_key(key)
            ctr = self.philox_2x_round(ctr, key)
        if r > 7:
            key = self.philox_2x_bump_key(key)
            ctr = self.philox_2x_round(ctr, key)
        if r > 8:
            key = self.philox_2x_bump_key(key)
            ctr = self.philox_2x_round(ctr, key)
        if r > 9:
            key = self.philox_2x_bump_key(key)
            ctr = self.philox_2x_round(ctr, key)
        if r > 10:
            key = self.philox_2x_bump_key(key)
            ctr = self.philox_2x_round(ctr, key)
        if r > 11:
            key = self.philox_2x_bump_key(key)
            ctr = self.philox_2x_round(ctr, key)
        if r > 12:
            key = self.philox_2x_bump_key(key)
            ctr = self.philox_2x_round(ctr, key)
        if r > 13:
            key = self.philox_2x_bump_key(key)
            ctr = self.philox_2x_round(ctr, key)
        if r > 14:
            key = self.philox_2x_bump_key(key)
            ctr = self.philox_2x_round(ctr, key)
        if r > 15:
            key = self.philox_2x_bump_key(key)
            ctr = self.philox_2x_round(ctr, key)
        return ctr

    def philox_2x(self, ctr, key, to_float=False):
        """Generate random values, with 10 as default number of rounds.

        :param ctr: counter array: uint64 vector
        :param key: key: uint32 scalar
        :param to_float: if False, return uint32 generated values as is.
            If true, convert them to float32 and return float32 values.
        :return: random values in uint32 or float32 vector with expected
            output size.
        """
        output_64 = self.impl_philox_2x_r(PHILOX2x32_DEFAULT_ROUNDS, ctr, key)
        output = self.__uint64_to_2xuint32_vector(output_64)
        if self.output_size % 2 == 1:
            output = relay.op.transform.strided_slice(
                output, [0], [2 * self.n - 1])
        if to_float:
            return self.__uint32_to_float32(output)
        return output
