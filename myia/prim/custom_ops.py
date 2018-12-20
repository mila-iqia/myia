"""user define operations.

User define operations list

"""

import numpy

def array_conv(bottom, height, width, kernel_size, stride, padding):
    
    h_size = (height+ 2*padding - kernel_size)//stride+1
    w_size = (width + 2*padding - kernel_size)//stride+1
    top = np.zeros(shape=(h_size*w_size,kernel_size*kernel_size))
    
    for h in range(h_size):
        for w in range(w_size):
            #print bottom[h*stride:h*stride+kernel_size,w*stride:w*stride+kernel_size]
            t=bottom[h*stride:h*stride+kernel_size,w*stride:w*stride+kernel_size]
            top[h*w_size+w]=t.reshape(1,kernel_size*kernel_size)
    return top
