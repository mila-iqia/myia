import numpy

def conv_op(bottom,height,width,kernel_size,stride,padding):
    h_size = (height+ 2*padding - kernel_size)//stride+1 #单维度上应该滑动的次数 即特征图的尺寸
    w_size = (width + 2*padding - kernel_size)//stride+1 #单维度上应该滑动的次数 即特征图的尺寸
    top = np.zeros(shape=(h_size*w_size,kernel_size*kernel_size))
    for h in range(h_size):#竖直方向第h次
        for w in range(w_size):#水平方向第w次
            #print bottom[h*stride:h*stride+kernel_size,w*stride:w*stride+kernel_size]
            t=bottom[h*stride:h*stride+kernel_size,w*stride:w*stride+kernel_size]
            top[h*w_size+w]=t.reshape(1,kernel_size*kernel_size)
    return top


