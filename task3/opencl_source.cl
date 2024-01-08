kernel void map(global float *in, global float *out) {
  const int globalId = get_global_id(0);
  out[globalId] = in[globalId] > 0 ? 1 : 0;
}

kernel void scan_partial(global float *in, global int *out) {
  const int globalId = get_global_id(0);
  const int localId = get_local_id(0);
  const int localSize = get_local_size(0);

  local float buff[1024];
  buff[localId] = in[globalId];
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
  out[globalId] = buff[localId];
}

kernel void scan_total(global int *in) {
  const int groupId = get_group_id(0);
  const int numGroups = get_num_groups(0);

  const int localId = get_local_id(0);
  const int localSize = get_local_size(0);
  if (groupId == 0) {
    for (int j = 1; j < numGroups; j++) {
      in[j * localSize + localId] += in[j * localSize - 1];
      barrier(CLK_GLOBAL_MEM_FENCE);
    }
  }
}

kernel void scatter(global float *in, global int *index, global float *out) {
  const int globalId = get_global_id(0);
  const int localId = get_local_id(0);
  const int localSize = get_local_size(0);

  local float inBuff[1024];
  inBuff[localId] = in[globalId];
  local int indexBuff[1024];
  indexBuff[localId] = index[globalId];
  barrier(CLK_LOCAL_MEM_FENCE);

  if (localId == 0) {
    if (globalId == 0 && indexBuff[0] > 0) {
      out[0] = inBuff[0];
    }
    if (globalId > 0 && index[globalId - 1] < indexBuff[0]) {
      out[indexBuff[0] - 1] = inBuff[0];
    }
  }

  if (localId > 0 && indexBuff[localId - 1] < indexBuff[localId]) {
    out[indexBuff[localId - 1]] = inBuff[localId];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  const int groupId = get_group_id(0);
  const int numGroups = get_num_groups(0);

  const int globalSize = get_global_size(0);
  if (globalId == globalSize - 1) {
    out[globalSize] = index[globalId];
  }
}