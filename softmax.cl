__kernel void vector_add(__global const float* A, __global const float* B, __global float* C, int n) {
    int id = get_global_id(0);
    if (id < n) {
        C[id] = A[id] + B[id];
    }
}

__kernel void max(__global const float *arr, __global size_t *dims, __global float *results, int numDims, int axis) {
    int id = get_global_id(0);
    
    ulong size = 1;
    for (int i = 0; i < numDims; i++) {
        if (i != axis) {
            size *= dims[i];
        }
    }
    if (id > size) {
        return;
    }

    ulong stride = 1;
    for (int i = numDims - 1; i > axis; i--) {
        stride *= dims[i];
    }

    size_t dimSizesUpTo = stride * dims[axis];
    size_t pos = (id / stride) * dimSizesUpTo + id % stride;

    float maxVal = FLT_MIN;
    for (size_t j = 0; j < dims[axis]; j++) {
        size_t offset = pos + j * stride;
        float val = arr[offset];
        if (val > maxVal) {
            maxVal = val;
        }
    }
    if (arr == results) {
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
    results[id] = maxVal;
    dims[axis] = 1;
}

__kernel void sum(__global const float *arr, __global size_t *dims, __global float *results, int numDims, int axis) {
    int id = get_global_id(0);
    
    ulong size = 1;
    for (int i = 0; i < numDims; i++) {
        if (i != axis) {
            size *= dims[i];
        }
    }
    if (id > size) {
        return;
    }

    ulong stride = 1;
    for (int i = numDims - 1; i > axis; i--) {
        stride *= dims[i];
    }

    size_t dimSizesUpTo = stride * dims[axis];
    size_t pos = (id / stride) * dimSizesUpTo + id % stride;

    float sum = 0.0f;
    for (size_t j = 0; j < dims[axis]; j++) {
        size_t offset = pos + j * stride;
        float val = arr[offset];
        sum += val;
    }
    // If we reuse the buffer we need a barrier
    if (arr == results) {
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
    results[id] = sum;
    dims[axis] = 1;
}

__kernel void sub_and_exp(__global float *arr, __global const size_t *dims,
                  __global const float *vals, __global const size_t *valDims,
                  __global const size_t *dimCntr, __global const size_t *valDimsCntr,
                  int numDims) {
    int id = get_global_id(0);
    
    int valPos = 0;
    for (int i = 0; i < numDims; i++) {
        if (valDims[i] != 1) {
            valPos += ( (id / dimCntr[i]) % dims[i] ) * valDimsCntr[i];
        }
    }
    arr[id] = exp(arr[id] - vals[valPos]);
}

__kernel void div(__global float *arr, __global const size_t *dims,
                  __global const float *vals, __global const size_t *valDims,
                  __global const size_t *dimCntr, __global const size_t *valDimsCntr,
                  int numDims) {
    int id = get_global_id(0);
    
    int valPos = 0;
    for (int i = 0; i < numDims; i++) {
        if (valDims[i] != 1) {
            valPos += ( (id / dimCntr[i]) % dims[i] ) * valDimsCntr[i];
        }
    }
    arr[id] = arr[id] / vals[valPos];
}

__kernel void mul(__global float *arr, float alpha) {
    int id = get_global_id(0);
    arr[id] *= alpha;
}