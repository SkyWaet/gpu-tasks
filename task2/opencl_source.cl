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

  int globalId = get_global_id(0);
  int globalSize = get_global_id(0);

  if (globalId == 0 && globalSize / localSize <= localSize) {
    float sum = 0;
    int numGroups = get_num_groups(0);

    for (int i = 0; i < numGroups; i++) {
      sum += result[i];
    }
    result[0] = sum;
  }
}

kernel void scan_inclusive(global float *a, global float *b,
                           global float *result) {
  // TODO: Implement OpenCL version.
}