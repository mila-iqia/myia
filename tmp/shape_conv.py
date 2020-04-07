import argparse

import numpy as np
import torch
import torch.nn.functional
import tvm
from tvm import relay

import conv2d_transpose as theano

CHECK_THEANO = False
CHECK_RELAY = False
CHECK_RELAY_GRAPH = False


def compile_relay_function(inputs, output):
    f = relay.Function(list(inputs), output)
    m = tvm.IRModule({})
    fn = relay.GlobalVar("f")
    m[fn] = f
    e = relay.create_executor(mod=m)
    c = e.evaluate(m[fn])
    return c


def ensure_couple(v):
    """
    If v is an integer, return (v, v)
    Else, check if va is a tuple or list of 2 integers
    """
    if isinstance(v, int):
        return v, v
    if isinstance(v, (tuple, list)):
        assert len(v) == 2
        assert all(isinstance(e, int) for e in v)
        return v
    raise ValueError("Expected a int or a couple of int, got %s" % v)


def assert_allclose(t1: np.ndarray, t2: np.ndarray, rtol=1e-5, atol=1e-8):

    # Quick debug code to display mismatching values
    shape = t1.shape
    if len(shape) > 1:
        x1 = t1.flatten()
        x2 = t2.flatten()
        s = x1.shape[0]
        c = 0
        for i in range(s):
            v1 = x1[i].item()
            v2 = x2[i].item()
            if abs(v1 - v2) > atol:
                c += 1
                print("diff\t%s\t%s\t%s\t%s" % (c, i + 1, v1, v2))
    # End debug code

    np.testing.assert_allclose(t1, t2, rtol=rtol, atol=atol, verbose=True)
    return True


class Conv2D:
    """
    Class to compute and check a conv2d output shape.
    """

    def __init__(
        self,
        n=0,
        c_in=0,
        h_in=0,
        w_in=0,
        kernel_size=0,
        c_out=0,
        strides=1,
        padding=0,
        dilation=1,
        groups=1,
    ):
        self.n = n
        self.c_in = c_in
        self.h_in = h_in
        self.w_in = w_in
        self.c_out = c_out
        self.strides = ensure_couple(strides)
        self.padding = ensure_couple(padding)
        self.dilation = ensure_couple(dilation)
        self.kernel_size = ensure_couple(kernel_size)
        self.groups = groups
        assert isinstance(self.n, int)
        assert isinstance(self.c_in, int)
        assert isinstance(self.h_in, int)
        assert isinstance(self.w_in, int)
        assert isinstance(self.c_out, int)
        assert isinstance(self.groups, int)
        assert self.c_out > 0
        assert self.groups > 0
        assert self.c_in % self.groups == 0
        assert self.c_out % self.groups == 0

    def debug_output(self):
        print(
            "%s %s filter=%s strides=%s, padding=%s, dilation=%s, groups=%s"
            % (
                type(self).__name__,
                self.get_input_shape(),
                self.get_filter_shape(),
                self.strides,
                self.padding,
                self.dilation,
                self.groups,
            )
        )
        output_shape = self.get_output_shape()
        # Use torch to check if output shape is computed correctly.
        torch_output = self.compute_torch()
        torch_output_shape = tuple(torch_output.shape)
        assert output_shape == torch_output_shape, (
            "%s: mismatch, expected %s, got %s"
            % (type(self).__name__, output_shape, torch_output_shape),
        )
        print(type(self).__name__, "OK", output_shape)
        print()

    def get_input_shape(self):
        return self.n, self.c_in, self.h_in, self.w_in

    def get_filter_shape(self):
        return (self.c_out, self.c_in // self.groups) + self.kernel_size

    def get_output_shape(self):
        n = self.n
        c_out = self.c_out
        h_out = int(
            (
                self.h_in
                + 2 * self.padding[0]
                - self.dilation[0] * (self.kernel_size[0] - 1)
                - 1
            )
            / self.strides[0]
            + 1
        )
        w_out = int(
            (
                self.w_in
                + 2 * self.padding[1]
                - self.dilation[1] * (self.kernel_size[1] - 1)
                - 1
            )
            / self.strides[1]
            + 1
        )
        return n, c_out, h_out, w_out

    def compute_torch(self):
        i = torch.randn(*self.get_input_shape())
        w = torch.randn(*self.get_filter_shape())
        return torch.nn.functional.conv2d(
            i,
            w,
            stride=self.strides,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def grad_input(self):
        """
        Infer and return parameters for conv2d input grad.
        """
        input_size = self.get_input_shape()
        weight_shape = self.get_filter_shape()
        grad_output_shape = self.get_output_shape()
        input_size = input_size[2:]
        min_sizes = [
            (grad_output_shape[i + 2] - 1) * self.strides[i]
            - 2 * self.padding[i]
            + (self.kernel_size[i] - 1) * self.dilation[i]
            + 1
            for i in range(2)
        ]
        grad_input_padding = [input_size[i] - min_sizes[i] for i in range(2)]
        return ConvTranspose2D(
            n=grad_output_shape[0],
            c_in=grad_output_shape[1],
            h_in=grad_output_shape[2],
            w_in=grad_output_shape[3],
            c_out=weight_shape[1] * self.groups,
            strides=self.strides,
            padding=self.padding,
            dilation=self.dilation,
            kernel_size=self.kernel_size,
            groups=self.groups,
            output_padding=grad_input_padding,
        )


class ConvTranspose2D:
    def __init__(
        self,
        n=0,
        c_in=0,
        h_in=0,
        w_in=0,
        kernel_size=0,
        c_out=0,
        strides=1,
        padding=0,
        dilation=1,
        groups=1,
        output_padding=0,
    ):
        self.n = n
        self.c_in = c_in
        self.h_in = h_in
        self.w_in = w_in
        self.c_out = c_out
        self.strides = ensure_couple(strides)
        self.padding = ensure_couple(padding)
        self.dilation = ensure_couple(dilation)
        self.kernel_size = ensure_couple(kernel_size)
        self.output_padding = ensure_couple(output_padding)
        self.groups = groups
        assert isinstance(self.n, int)
        assert isinstance(self.c_in, int)
        assert isinstance(self.h_in, int)
        assert isinstance(self.w_in, int)
        assert isinstance(self.c_out, int)
        assert isinstance(self.groups, int)
        assert self.c_out > 0
        assert self.groups > 0
        assert self.c_in % self.groups == 0
        assert self.c_out % self.groups == 0

    def debug_output(self):
        print(
            "%s %s filter=%s strides=%s, padding=%s, dilation=%s, groups=%s, output_padding=%s"
            % (
                type(self).__name__,
                self.get_input_shape(),
                self.get_filter_shape(),
                self.strides,
                self.padding,
                self.dilation,
                self.groups,
                self.output_padding,
            )
        )

        output_shape = self.get_output_shape()
        i, w = self._inputs()
        # Use torch to check if output shape is computed correctly.
        # Torch output will be also used to check other implementations below.
        torch_output = self.compute_torch(i, w)
        torch_output_shape = tuple(torch_output.shape)
        assert output_shape == torch_output_shape, (
            "%s: mismatch, expected %s, got %s"
            % (type(self).__name__, output_shape, torch_output_shape),
        )
        print(type(self).__name__, "OK", output_shape)

        if CHECK_THEANO:
            # Check theano implementation.
            self._debug_theano(i, w, torch_output)

        if CHECK_RELAY:
            # Check relay function conv2d_transpose.
            self._debug_relay(i, w, torch_output)

        if CHECK_RELAY_GRAPH:
            # Check relay graph implementation.
            self._debug_relay_graph(i, w, torch_output)

        print()

    def _debug_theano(self, i, w, expected):
        output_shape = tuple(expected.shape)
        theano_output = self.compute_theano(i, w)
        theano_output_shape = tuple(theano_output.shape)
        assert output_shape == theano_output_shape, (
            "Theano %s: mismatch, expected %s, got %s"
            % (type(self).__name__, output_shape, theano_output_shape),
        )
        assert_allclose(theano_output, expected, atol=1e-5)
        print("Theano", type(self).__name__, "OK", output_shape)

    def _debug_relay(self, i, w, expected):
        output_shape = tuple(expected.shape)
        relay_output = self.compute_relay(i, w)
        relay_output_shape = tuple(relay_output.shape)
        assert output_shape == relay_output_shape, (
            "Relay %s: mismatch, expected %s, got %s"
            % (type(self).__name__, output_shape, relay_output_shape),
        )
        assert_allclose(relay_output, expected, atol=1e-5)
        print("Relay", type(self).__name__, "OK", output_shape)

    def _debug_relay_graph(self, i, w, expected):
        output_shape = tuple(expected.shape)
        relay_graph_output = self.compute_relay_using_conv2d(i, w)
        if any(p != 0 for p in self.padding):
            # Small more step, will be integrated to relay graph later
            # (once bugs will be fixed)
            relay_graph_output = relay_graph_output[
                :,
                :,
                self.padding[0] : -self.padding[0],
                self.padding[1] : -self.padding[1],
            ]
        relay_graph_output_shape = tuple(relay_graph_output.shape)
        assert output_shape == relay_graph_output_shape, (
            "Relay Graph %s: mismatch, expected %s, got %s"
            % (type(self).__name__, output_shape, relay_graph_output_shape),
        )
        assert_allclose(relay_graph_output, expected, atol=1e-5)
        print("Relay Graph", type(self).__name__, "OK", output_shape)

    def get_input_shape(self):
        return self.n, self.c_in, self.h_in, self.w_in

    def get_filter_shape(self):
        return (self.c_in, self.c_out // self.groups) + self.kernel_size

    def get_output_shape(self):
        n = self.n
        c_out = self.c_out
        h_out = (
            (self.h_in - 1) * self.strides[0]
            - 2 * self.padding[0]
            + self.dilation[0] * (self.kernel_size[0] - 1)
            + self.output_padding[0]
            + 1
        )
        w_out = (
            (self.w_in - 1) * self.strides[1]
            - 2 * self.padding[1]
            + self.dilation[1] * (self.kernel_size[1] - 1)
            + self.output_padding[1]
            + 1
        )
        return n, c_out, h_out, w_out

    def compute_torch(self, i, w):
        i = torch.as_tensor(i)
        w = torch.as_tensor(w)
        return torch.nn.functional.conv_transpose2d(
            i,
            w,
            stride=self.strides,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
        ).numpy()

    def compute_theano(self, i, w):
        return theano.conv2d_transpose(
            i,
            w,
            self.output_padding,
            padding=self.padding,
            strides=self.strides,
            dilation=self.dilation,
            groups=self.groups,
        )

    def compute_relay(self, i, w):
        if self.groups != 1 or self.dilation != (1, 1):
            return None
        if any(op > 1 for op in self.output_padding):
            return None
        data = relay.var("data", shape=self.get_input_shape())
        weight = relay.var("weight", shape=self.get_filter_shape())
        o = relay.nn.conv2d_transpose(
            data,
            weight,
            strides=self.strides,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            channels=self.c_out,
            output_padding=self.output_padding,
        )
        f = compile_relay_function([data, weight], o)
        return f(i, w).asnumpy()

    def compute_relay_using_conv2d(self, i, w):
        """
        Implementation of conv2d_transpose using relay conv2d function.
        NB: Need operation relay.nn.dilate. I made a branch
        to add it here (not yet submitted):
        https://github.com/notoraptor/incubator-tvm/tree/relay-op-dilate
        """

        osh = self.get_output_shape()
        topgrad_shape = self.get_input_shape()
        kern_shape = self.get_filter_shape()

        data = relay.var("data", shape=topgrad_shape)
        weight = relay.var("weight", shape=kern_shape)
        d1 = relay.nn.dilate(data, (1, 1) + self.strides)
        work = relay.nn.pad(
            d1,
            (
                (0, 0),
                (0, 0),
                (0, self.output_padding[0]),
                (0, self.output_padding[1]),
            ),
        )
        kern, kshp = self._relay_preprocess_kernel(weight, kern_shape)
        conv_padding = tuple(
            (kshp[-2 + i] - 1) * self.dilation[i] for i in range(2)
        )
        _work = relay.nn.pad(
            work,
            ((0, 0), (0, 0), (conv_padding[0],) * 2, (conv_padding[1],) * 2),
        )
        _kern = relay.nn.dilate(kern, (1, 1) + self.dilation)
        img = relay.nn.conv2d(
            _work, _kern, groups=self.groups, channels=osh[1],
        )

        # Temporarly deactivated.

        # if any(p != 0 for p in self.p):
        #     img = relay.op.transform.strided_slice(
        #         data=img,
        #         begin=[0, 0, self.p[0], self.p[1]],
        #         end=[None, None, osh[2] + self.p[0], osh[3] + self.p[1]],
        #     )

        f = compile_relay_function([data, weight], img)
        return f(i, w).asnumpy()

    def _relay_preprocess_kernel(self, mat, mat_shape):
        mshp0 = mat_shape[0] // self.groups
        mshp1 = mat_shape[1] * self.groups
        mat = relay.reshape(mat, (self.groups, mshp0) + mat_shape[1:])
        # => (self.g, mshp0, m1, m2, m3)
        mat = relay.op.transpose(mat, axes=(1, 0, 2, 3, 4))
        # => (mshp0, self.g, m1, m2, m3)
        mat = relay.reshape(mat, (mshp0, mshp1, mat_shape[-2], mat_shape[-1]))
        # => (mshp0, mshp1, m2, m3)
        mat = relay.op.transpose(mat, (1, 0, 2, 3))
        # => (mshp1, mshp0, m2, m3)
        # Kernel must be flipped
        mat = relay.op.transform.reverse(mat, 2)
        mat = relay.op.transform.reverse(mat, 3)
        return mat, (mshp1, mshp0, mat_shape[2], mat_shape[3])

    def _inputs(self):
        i = np.random.rand(*self.get_input_shape()).astype("float32")
        w = np.random.rand(*self.get_filter_shape()).astype("float32")
        return i, w


def tests():
    # Check conv
    c = Conv2D(
        n=2,
        c_in=6,
        h_in=4,
        w_in=5,
        c_out=6,
        strides=(2, 3),
        padding=(3, 2),
        dilation=(1, 1),
        kernel_size=(3, 3),
        groups=1,
    )
    c.debug_output()
    # Check grad input
    d1 = c.grad_input()
    d1.debug_output()
    # Get grad input, change some parameters, and check it
    d2 = c.grad_input()
    d2.output_padding = (2, 0)
    d2.strides = (4, 3)
    d2.debug_output()

    # Check conv and grad input for each value
    for conv in (
        Conv2D(
            n=3,
            c_in=12,
            h_in=60,
            w_in=80,
            c_out=3,
            strides=(2, 3),
            padding=(2, 2),
            dilation=(1, 1),
            kernel_size=(5, 5),
            groups=3,
        ),
        Conv2D(
            n=3,
            c_in=12,
            h_in=60,
            w_in=80,
            c_out=3,
            strides=(2, 3),
            padding=(2, 2),
            dilation=(1, 2),
            kernel_size=(5, 5),
            groups=3,
        ),
        Conv2D(
            n=3,
            c_in=12,
            h_in=60,
            w_in=80,
            c_out=3,
            strides=(2, 3),
            padding=0,
            dilation=(1, 1),
            kernel_size=(5, 5),
            groups=3,
        ),
        Conv2D(
            n=3,
            c_in=12,
            h_in=60,
            w_in=80,
            c_out=3,
            strides=1,
            padding=(2, 2),
            dilation=(1, 2),
            kernel_size=(5, 5),
            groups=3,
        ),
        Conv2D(
            n=3,
            c_in=3,
            h_in=60,
            w_in=80,
            c_out=1,
            strides=(2, 3),
            padding=(2, 2),
            dilation=(1, 1),
            kernel_size=(5, 5),
            groups=1,
        ),
        Conv2D(
            n=3,
            c_in=3,
            h_in=60,
            w_in=80,
            c_out=1,
            strides=(2, 3),
            padding=(2, 2),
            dilation=(2, 1),
            kernel_size=(5, 5),
            groups=1,
        ),
        Conv2D(
            n=3,
            c_in=3,
            h_in=60,
            w_in=80,
            c_out=1,
            strides=(2, 3),
            padding=0,
            dilation=(1, 1),
            kernel_size=(5, 5),
            groups=1,
        ),
        Conv2D(
            n=3,
            c_in=3,
            h_in=60,
            w_in=80,
            c_out=6,
            strides=(2, 3),
            padding=(2, 2),
            dilation=(1, 1),
            kernel_size=(5, 5),
            groups=3,
        ),
        Conv2D(
            n=3,
            c_in=3,
            h_in=60,
            w_in=80,
            c_out=6,
            strides=(2, 3),
            padding=(2, 2),
            dilation=(2, 3),
            kernel_size=(5, 5),
            groups=3,
        ),
        Conv2D(
            n=3,
            c_in=3,
            h_in=60,
            w_in=80,
            c_out=6,
            strides=1,
            padding=(2, 2),
            dilation=(2, 3),
            kernel_size=(5, 5),
            groups=3,
        ),
    ):
        conv.debug_output()
        conv.grad_input().debug_output()

    # Check conv transpose 2D directly
    ConvTranspose2D(
        n=2,
        c_in=6,
        h_in=4,
        w_in=5,
        c_out=2,
        kernel_size=(3, 3),
        strides=(2, 3),
        padding=(3, 2),
        dilation=(1, 1),
        groups=1,
    ).debug_output()
    ConvTranspose2D(
        n=2,
        c_in=6,
        h_in=4,
        w_in=5,
        c_out=2,
        kernel_size=(3, 3),
        strides=(2, 3),
        padding=(3, 2),
        dilation=(1, 1),
        groups=1,
    ).debug_output()


def main():
    global CHECK_THEANO, CHECK_RELAY, CHECK_RELAY_GRAPH
    parser = argparse.ArgumentParser(
        description="Check conv2d shapes and conv transpose 2d sahpes and implementations."
    )
    parser.add_argument(
        "--theano",
        "-t",
        action="store_true",
        default=False,
        help="Check Theano-based implementation for conv transpose 2D (default False)",
    )
    parser.add_argument(
        "--relay",
        "-r",
        action="store_true",
        default=False,
        help="Check Relay function conv2d_tranapose (default False)",
    )
    parser.add_argument(
        "--relay-graph",
        "-g",
        action="store_true",
        default=False,
        help="Check Relay graph implementation for conv transpose 2D (default False)",
    )
    args = parser.parse_args()
    CHECK_THEANO = args.theano
    CHECK_RELAY = args.relay
    CHECK_RELAY_GRAPH = args.relay_graph
    tests()


if __name__ == "__main__":
    main()
