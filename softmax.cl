
size_t next_power_of_2(size_t n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    n++;
    return n;
}

size_t get_out_pos(__global size_t *dims, const uint local_size, 
               const size_t dimension, const size_t part,
               const uint numDims, const uint axis) {
    size_t newDimSize = next_power_of_2(dims[axis]) / local_size;
    size_t outPos = 0;
    size_t redDimCtr = 1;
    size_t dimCtr = 1;

    for (int i = numDims-1; i >= 0; i--) {
        size_t dim = 1;
        if (i == axis) {
            dim = newDimSize;
            outPos += part * dimCtr;
        } else {
            dim = dims[i];
            outPos += (dimension / redDimCtr) % dim * dimCtr;
            redDimCtr *= dim;
        }
        dimCtr *= dim;
    }

    return outPos;
}

__kernel void reduce_sum_ND(__local float* cache, __global float* input, __global size_t *inDims,
                            __global float* output, __global size_t *outDims,
                            const uint N, const uint numDims, const uint axis)
{
    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);
    const uint group_id = get_group_id(0);
    const uint local_size = get_local_size(0);

    const uint num_groups = get_global_size(0) / local_size;

    if (global_id == 0) {
        for (int i = 0; i < numDims; i++) {
            if (i == axis) {
                outDims[axis] = next_power_of_2(inDims[i]) / (local_size);
            } else {
                outDims[i] = inDims[i];
            }
        }
    }

    ulong stride = 1;
    for (int i = numDims - 1; i > axis; i--) {
        stride *= inDims[i];
    }


    size_t partSize = next_power_of_2(inDims[axis]) / local_size;
    size_t partNo = group_id % partSize;
    size_t dimension = group_id / partSize;

    size_t dimSizesUpTo = stride * inDims[axis];
    size_t startPos = (dimension / stride) * dimSizesUpTo + dimension % stride;

    size_t localOffsetPos = partNo * local_size + local_id;
    size_t offset = startPos + localOffsetPos * stride;
    cache[local_id] = (localOffsetPos < inDims[axis]) ? input[offset] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int s = local_size >> 1; s > 0; s >>= 1) {
        if (local_id < s) {
            cache[local_id] += cache[local_id + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0 && group_id < N) { 
        size_t outPos = get_out_pos(inDims, local_size, dimension, partNo, numDims, axis);
        output[outPos] = cache[0];
    }
}

__kernel void reduce_max_ND(__local float* cache, __global float* input, __global size_t *inDims,
                            __global float* output, __global size_t *outDims,
                            const uint N, const uint numDims, const uint axis)
{
    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);
    const uint group_id = get_group_id(0);
    const uint local_size = get_local_size(0);

    const uint num_groups = get_global_size(0) / local_size;

    if (global_id == 0) {
        for (int i = 0; i < numDims; i++) {
            if (i == axis) {
                outDims[axis] = next_power_of_2(inDims[i]) / (local_size);
            } else {
                outDims[i] = inDims[i];
            }
        }
    }

    ulong stride = 1;
    for (int i = numDims - 1; i > axis; i--) {
        stride *= inDims[i];
    }

    size_t partSize = next_power_of_2(inDims[axis]) / local_size;
    size_t partNo = group_id % partSize;
    size_t dimension = group_id / partSize;

    size_t dimSizesUpTo = stride * inDims[axis];
    size_t startPos = (dimension / stride) * dimSizesUpTo + dimension % stride;

    size_t localOffsetPos = partNo * local_size + local_id;
    size_t offset = startPos + localOffsetPos * stride;
    cache[local_id] = (localOffsetPos < inDims[axis]) ? input[offset] : FLT_MIN;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int s = local_size >> 1; s > 0; s >>= 1) {
        if (local_id < s) {
            cache[local_id] = (cache[local_id] > cache[local_id + s]) ?
                               cache[local_id] :
                               cache[local_id + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0 && group_id < N) { 
        size_t outPos = get_out_pos(inDims, local_size, dimension, partNo, numDims, axis);
        output[outPos] = cache[0];
    }
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