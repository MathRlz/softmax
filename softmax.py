import numpy as np

def softmax(x, alpha = 1.0, axis = None):

    y = np.atleast_2d(x)

    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    y = y * float(alpha)

    print(f'Max: {np.max(y, axis = axis)}')
    print(f'Max expand {np.expand_dims(np.max(y, axis = axis), axis)}')
    print(f'Y before: {y}')
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    print(f'Y after: {y}')

    y = np.exp(y)
    print(f'y exp: {y}')

    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    print(f'ax_sum: {ax_sum}')

    p = y / ax_sum
    print(f'result: {p}')

    if len(x.shape) == 1: p = p.flatten()

    return p

test = np.array([[[1, 2, 3],
                                         [4, 5, 6],
                                         [7, 8, 9]],
                                       
                                        [[10, 11, 12],
                                         [13, 14, 15],
                                         [16, 17, 18]],
])
print(test)


result = softmax(test, 1.0, (1, 2))

