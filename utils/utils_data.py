import numpy as np


def ivecs_write(fname, m):
    n, d = m.shape
    print(f"gnd shape: {n}, {d}")
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    print(m1)
    m1.tofile(fname)


def fvecs_write(fname, m):
    m = m.astype('float32')
    ivecs_write(fname, m.view('int32'))


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    # print(a.dtype)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')