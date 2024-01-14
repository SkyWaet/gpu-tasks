kernel void map(global float *in, global int *out) {
  const int localSize = get_local_size(0);
  int groupId = get_group_id(0);
  int localId = get_local_id(0);
  local float buff[1024];

  buff[localId] = in[groupId * localSize + localId];
  barrier(CLK_LOCAL_MEM_FENCE);

  if (localId == 0) {
    int cnt = 0;
    for (int j = 0; j < localSize; j++) {
      if (buff[j] > 0)
        cnt++;
    }
    out[groupId] = cnt;
  }
}
kernel void scan_partial(global int *out) {
  const int globalId = get_global_id(0);
  const int localId = get_local_id(0);
  const int localSize = get_local_size(0);

  local int buff[1024];
  buff[localId] = out[globalId];
  barrier(CLK_LOCAL_MEM_FENCE);

  int sum = buff[localId];
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
  const int globalSize = get_global_size(0);
  const int localId = get_local_id(0);
  const int localSize = get_local_size(0);
  if (groupId == 0) {
    for (int j = 1; j < globalSize / localSize; j++) {
      in[j * localSize + localId] += in[j * localSize - 1];
      barrier(CLK_GLOBAL_MEM_FENCE);
    }
  }
}

kernel void scatter(global float *in, global int *offsets,
                    global float *out) {
  const int localSize = get_local_size(0);
  int groupId = get_group_id(0);
  int localId = get_local_id(0);
  local float buff[1024];

  buff[localId] = in[groupId * localSize + localId];
  barrier(CLK_LOCAL_MEM_FENCE);

  if (localId == 0) {
    int currentIndex = 0;
    if (groupId > 0)
      currentIndex = offsets[groupId - 1];
    for (int j = 0; j < localSize; j++) {
      if (buff[j] > 0) {
        out[currentIndex] = buff[j];
        currentIndex++;
      }
    }
  }
}