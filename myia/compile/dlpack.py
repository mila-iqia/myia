from ctypes import c_uint64, c_int64, c_uint, c_int, c_uint16, c_uint8, \
    c_void_p, Structure, POINTER, CFUNCTYPE

c_int64_p = POINTER(c_int64)

DLDeviceType = c_uint
kDLCPU = c_uint(1)
kDLGPU = c_uint(2)
kDLCPUPinned = c_uint(3)
kDLOpenCL = c_uint(4)
kDLVulkan = c_uint(7)
kDLMetal = c_uint(8)
kDLVPI = c_uint(9)
kDLROCM = c_uint(10)
kDLExtDev = c_uint(12)


class DLContext(Structure):
    _fields_ = [('device_type', DLDeviceType),
                ('device_id', c_int)]


DLDataTypeCode = c_uint
kDLInt = c_uint(0)
kDLUInt = c_uint(1)
kDLFloat = c_uint(2)


class DLDataType(Structure):
    _fields_ = [('code', c_uint8),
                ('bits', c_uint8),
                ('lanes', c_uint16)]


class DLTensor(Structure):
    _fields_ = [('data', c_void_p),
		('ctx', DLContext),
		('ndim', c_int),
                ('dtype', DLDataType),
                ('shape', c_int64_p),
		('strides', c_int64_p),
		('byte_offset', c_uint64)]


class DLManagedTensor(Structure):
    pass


DLManagedTensor._fields_ = [
    ('dl_tensor', DLTensor),
    ('manager_ctx', c_void_p),
    ('deleter', CFUNCTYPE(None, POINTER(DLManagedTensor)))
]
