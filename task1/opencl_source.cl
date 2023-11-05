
kernel void vector_times_vector(global float *a, global float *b,
                                global float *result) {
  const int i = get_global_id(0);
  result[i] = a[i] * b[i];
}

kernel void matrix_times_vector(global const float *a, global const float *b,
                                global float *result) {
  const int i = get_global_id(0);
  const int n = get_global_size(0);
  float sum = 0;
  for (int j = 0; j < n; ++j) {
    sum += a[i * n + j] * b[j];
  }
  result[i] = sum;
}

kernel void matrix_times_matrix(global float *a, global float *b,
                                global float *result) {
  int tx = get_global_id(0);
  int ty = get_global_id(1);
  int wA = get_global_size(0);

  int localSize = get_local_size(0);
  int localId = get_local_id(0);

  local float rowBuff[1024];

  for (int i = localId; i < wA; i += localSize) {
    rowBuff[i] = a[ty * wA + i];
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  float value = 0;
  for (int k = 0; k < wA; ++k) {
    float elementB = b[k * wA + tx];
    value += rowBuff[k] * elementB;
  }
  result[ty * wA + tx] = value;
}