__kernel void vec_add(__global uint *in_a, __global uint *in_b,
                      __global uint *out) {
  size_t idx = get_global_id(0);

  out[idx] = in_a[idx] + in_b[idx];
}

// Vectorized modular multiplication for prime field
// F(2 ** 64 - 2 ** 32 + 1)
//
// This function is re-implementation of
// https://github.com/itzmeanjan/ff-gpu/blob/7ad664cff8713b5e6bfe5527531a7532e33abd47/ff_p.cpp#L44-L70
// where I performed same modular multiplication of two 64-bit prime field
// elements, here I extend that to work on vector of length 16
ulong16 vec_mul_ff_p64(ulong16 a, ulong16 b) {
  ulong16 ab = a * b;
  ulong16 cd = mul_hi(a, b);
  ulong16 c = cd & 0x00000000ffffffff;
  ulong16 d = cd >> 32;

  ulong16 tmp_0 = ab - d;
  long16 under_0 = ab < d;
  ulong16 tmp_1 = as_ulong16(under_0) & 0x00000000ffffffff;
  ulong16 tmp_2 = tmp_0 - tmp_1;

  ulong16 tmp_3 = (c << 32) - c;

  ulong16 tmp_4 = tmp_2 + tmp_3;
  long16 over_0 = tmp_2 > (ULONG_MAX - tmp_3);
  ulong16 tmp_5 = as_ulong16(over_0) & 0x00000000ffffffff;

  return tmp_4 + tmp_5;
}
