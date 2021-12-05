__kernel void vec_add(__global uint *in_a, __global uint *in_b,
                      __global uint *out) {
  size_t idx = get_global_id(0);

  out[idx] = in_a[idx] + in_b[idx];
}

constant ulong MOD = 18446744069414584321ul;

// Vectorized modular multiplication for prime field
// F(2 ** 64 - 2 ** 32 + 1) = F(MOD)
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

// Vectorized modular addition of 16 prime field elements
// where prime field modulo is MOD
//
// This function is adapted implementation of scalar modular addition
// routine
// https://github.com/itzmeanjan/ff-gpu/blob/7ad664cff8713b5e6bfe5527531a7532e33abd47/ff_p.cpp#L4-L22
ulong16 vec_add_ff_p64(ulong16 a, ulong16 b) {
  // instead of doing b % MOD, I'm executing
  // next 4 instructions
  //
  // So expectation is assert(b_ok == b % mod)
  // must pass
  ulong16 mod_vec = (ulong16)(MOD);
  long16 over_0 = b >= MOD;
  ulong16 tmp_0 = (as_ulong16(over_0) >> 63) * mod_vec;
  ulong16 b_ok = b - tmp_0;

  ulong16 tmp_1 = a + b_ok;
  long16 over_1 = a > (ULONG_MAX - b_ok);
  ulong16 tmp_2 = as_ulong16(over_1) & 0x00000000ffffffff;

  ulong16 tmp_3 = tmp_1 + tmp_2;
  long16 over_2 = tmp_1 > (ULONG_MAX - tmp_2);
  ulong16 tmp_4 = as_ulong16(over_2) & 0x00000000ffffffff;

  return tmp_3 + tmp_4;
}

// Function of raising each element of state vector to 7th power
// by performing multiple vector multiplications
//
// This routine is vectorization attempt of function operating on
// rescue prime hash function's state ( actually it permutes hash state )
//
// I adapted it from
// https://github.com/itzmeanjan/ff-gpu/blob/9c57cb13e4b2d96a084da96d558fe3d4707bfcb7/rescue_prime.cpp#L43-L50
ulong16 apply_sbox(ulong16 state) {
  // element-wise multiplication of vectors, so I've {a ^ 2 ∀ a ∈ state}
  ulong16 state2 = state * state;
  // element-wise multiplication of vectors, so I've {a ^ 4 ∀ a ∈ state}
  ulong16 state4 = state2 * state2;
  // element-wise multiplication of vectors, so I've {a ^ 6 ∀ a ∈ state}
  ulong16 state6 = state2 * state4;

  // element-wise multiplication of vectors, so I've {a ^ 7 ∀ a ∈ state}
  return state6 * state;
}
