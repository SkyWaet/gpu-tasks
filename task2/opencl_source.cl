kernel void reduce(global float *a, global float *result) {

  int localSize = get_local_size(0);
  int localId = get_local_id(0);

  int groupId = get_group_id(0);
  local float buff[1024];

  buff[localId] = a[groupId * localSize + localId];
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int offset = localSize / 2; offset > 0; offset /= 2) {
    if (localId < offset) {
      buff[localId] += buff[localId + offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (localId == 0) {
    result[groupId] = buff[0];
  }

  const int globalSize = get_global_size(0);
  const int globalId = get_global_id(0);

  if (globalId == 0 && globalSize / localSize <= localSize) {
    float sum = 0;
    const int numGroups = get_num_groups(0);
    for (int j = 0; j < numGroups; j++)
      sum += result[j];
    result[0] = sum;
  }
}

kernel void scan_inclusive(global float *a, int step) {
    const int globalId = get_global_id(0);
    const int localId = get_local_id(0);
    const int localSize = get_local_size(0);

    local float buff[1024];
    buff[localId] = a[(step - 1) + globalId * step];
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = buff[localId];
    for (int offset = 1; offset < localSize; offset *= 2) {
        if (localId >= offset) {
            sum += buff[localId - offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        buff[localId] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    a[(step - 1) + globalId * step] = buff[localId];
}

kernel void scan_inclusive_end(global float* a, int step) {
    const int globalId = get_global_id(0);
    for (int j = 0; j < step - 1; j++) {
        a[(globalId + 1) * step + j] += a[(globalId + 1) * step - 1];
    }
}