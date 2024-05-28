import numpy as np

def softmax(x, alpha = 1.0, axis = None):

    y = np.atleast_2d(x)

    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    y = y * float(alpha)

    print(f'Max: {np.max(y, axis = axis)}')
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    y = np.exp(y)

    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    p = y / ax_sum

    if len(x.shape) == 1: p = p.flatten()

    return p

x = np.array([[1,2], [3, 4]])

output = softmax(x, 1.0, (1))
print(output)

test = np.random.rand(2, 3)
print(test)
reduction_axes = (0)
print(np.max(test, axis=(0)))
print(np.max(test, axis=(1)))
print(np.max(test, axis=(0,1)))


test2 = np.array([[[1, 2, 3],
                                         [4, 5, 6],
                                         [7, 8, 9]],
                                       
                                        [[10, 11, 12],
                                         [13, 14, 15],
                                         [16, 17, 18]],
                                       
                                        [[19, 20, 21],
                                         [22, 23, 24],
                                         [25, 26, 27]]])
print(np.max(test2, axis=(0)))
print(np.max(test2, axis=(1)))
print(np.max(test2, axis=(2)))