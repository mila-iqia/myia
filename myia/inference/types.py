
# TODO: use the typing module instead. Int8 etc. would have to be defined
# as types in order to appear in typing.Tuple and so on.


class Type:
    def __init__(self, name, elem_types=None):
        self.name = name
        self.elem_types = elem_types

    def __getitem__(self, elem_types):
        assert self.elem_types == ()
        if not isinstance(elem_types, tuple):
            elem_types = (elem_types,)
        return Type(self.name, elem_types)

    def __hash__(self):
        return hash(self.name) ^ hash(self.elem_types)

    def __eq__(self, other):
        return isinstance(other, Type) \
            and other.name == self.name \
            and other.elem_types == self.elem_types

    def __str__(self):
        if self.elem_types:
            etypes = ", ".join(map(str, self.elem_types))
            return f'{self.name}[{etypes}]'
        else:
            return self.name

    def __repr__(self):
        return str(self)


Bool = Type('Bool')
Float32 = Type('Float32')
Float64 = Type('Float64')
Int8 = Type('Int8')
Int16 = Type('Int16')
Int32 = Type('Int32')
Int64 = Type('Int64')
UInt8 = Type('UInt8')
UInt16 = Type('UInt16')
UInt32 = Type('UInt32')
UInt64 = Type('UInt64')
Array = Type('Array', ())
Tuple = Type('Tuple', ())


type_map = {
    bool: Bool,
    float: Float64,
    int: Int64,
    'float32': Float32,
    'float64': Float64,
    'int8': Int8,
    'int16': Int16,
    'int32': Int32,
    'int64': Int64,
    'uint8': Int8,
    'uint16': Int16,
    'uint32': Int32,
    'uint64': Int64
}


def typeof(x):
    t = type(x)
    res = type_map.get(type(x), None)
    if res:
        return res
    if t.__name__ == 'ndarray' and hasattr(x, 'dtype'):
        return Array[type_map.get(x.dtype.name)]
    raise TypeError(f'Unknown data type: {type(x)}')
