"""user define operations.

User define operations list

"""
import numpy as np

def array_conv(bottom,height,width,kernel_size,stride):
    padding = 0
    h_size = (height+ 2*padding - kernel_size)//stride+1
    w_size = (width + 2*padding - kernel_size)//stride+1
    top = np.zeros(shape=(h_size*w_size,kernel_size*kernel_size))
    for h in range(h_size):
        for w in range(w_size):
            t=bottom[h*stride:h*stride+kernel_size,w*stride:w*stride+kernel_size]
            top[h*w_size+w]=t.reshape(1,kernel_size*kernel_size)
    return top

def cal_conv_grad(dout, height, width, kernel_size, stride):
    h_size = (height - kernel_size)//stride+1
    w_size = (width - kernel_size)//stride+1
    dx = np.zeros(shape=(height,width))
    for h in range(h_size):
        for w in range(w_size):
            t=dout[h * w_size + w]
            db = t.reshape(kernel_size, kernel_size)
            dx[h:h + 1, w:w + 1] = dx[h:h + 1, w:w + 1] + db
    return dx
