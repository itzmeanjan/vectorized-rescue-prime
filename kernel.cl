__kernel void vec_add(__global uint *in_a, __global uint *in_b,
                      __global uint *out) {
  size_t idx = get_global_id(0);

  out[idx] = in_a[idx] + in_b[idx];
}
