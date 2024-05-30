__kernel void reduce_sum_ND(__local float* cache, __global float* input, __global size_t *inDims,
                            __global float* output, __global size_t *outDims, const uint N, const uint numDims, int axis)
{
    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);
    const uint group_id = get_group_id(0);
    const uint local_size = get_local_size(0);

    const uint num_groups = get_global_size(0) / local_size;

    inDims[axis] /= num_groups;

    cache[local_id] = (global_id < N) ? input[global_id] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int s = local_size >> 1; s > 0; s >>= 1) {
        if (local_id < s) {
            cache[local_id] += cache[local_id + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) output[group_id] = cache[0];
}

__kernel void reduce_sum(__local float* cache, __global float* input, __global float* output, const unsigned int N)
{
    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);
    const uint group_id = get_group_id(0);
    const uint local_size = get_local_size(0);

    cache[local_id] = (global_id < N) ? input[global_id] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int s = local_size >> 1; s > 0; s >>= 1) {
        if (local_id < s) {
            cache[local_id] += cache[local_id + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) output[group_id] = cache[0];
}

__kernel void reduce_max(__local float* cache, __global float* input, __global float* output, const unsigned int N)
{
    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);
    const uint group_id = get_group_id(0);
    const uint local_size = get_local_size(0);

    cache[local_id] = (global_id < N) ? input[global_id] : FLT_MIN;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int s = local_size >> 1; s > 0; s >>= 1) {
        if (local_id < s) {
            cache[local_id] = (cache[local_id] > cache[local_id + s]) ? cache[local_id] : cache[local_id + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) output[group_id] = cache[0];
}

__kernel void max(__global const float *arr, __global size_t *dims, __global float *results, int numDims, int axis) {
    size_t id = get_global_id(0);
    
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
    size_t id = get_global_id(0);
    
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
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
    results[id] = sum;
    dims[axis] = 1;
}

__kernel void sub_and_exp(__global float *arr, __global const size_t *dims,
                  __global const float *vals, __global const size_t *valDims,
                  __global const size_t *dimCntr, __global const size_t *valDimsCntr,
                  int numDims) {
    size_t id = get_global_id(0);
    
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
    size_t id = get_global_id(0);
    
    int valPos = 0;
    for (int i = 0; i < numDims; i++) {
        if (valDims[i] != 1) {
            valPos += ( (id / dimCntr[i]) % dims[i] ) * valDimsCntr[i];
        }
    }
    arr[id] = arr[id] / vals[valPos];
}

__kernel void mul(__global float *arr, float alpha) {
    size_t id = get_global_id(0);
    arr[id] *= alpha;
}