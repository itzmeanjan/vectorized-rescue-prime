__kernel void vec_add(__global uint *in_a, __global uint *in_b,
                      __global uint *out) {
  size_t idx = get_global_id(0);

  out[idx] = in_a[idx] + in_b[idx];
}

constant ulong MOD = 18446744069414584321ul;
constant ulong STATE_WIDTH = 12ul;

// Vectorized modular multiplication for prime field
// F(2 ** 64 - 2 ** 32 + 1) = F(MOD)
//
// This function is re-implementation of
// https://github.com/itzmeanjan/ff-gpu/blob/7ad664cff8713b5e6bfe5527531a7532e33abd47/ff_p.cpp#L44-L70
// where I performed same modular multiplication of two 64-bit prime field
// elements, here I extend that to work on vector of length 16
inline ulong16 vec_mul_ff_p64(ulong16 a, ulong16 b) {
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
inline ulong16 vec_add_ff_p64(ulong16 a, ulong16 b) {
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
inline ulong16 apply_sbox(ulong16 state) {
  // element-wise multiplication of vectors, so I've {a ^ 2 ∀ a ∈ state}
  ulong16 state2 = state * state;
  // element-wise multiplication of vectors, so I've {a ^ 4 ∀ a ∈ state}
  ulong16 state4 = state2 * state2;
  // element-wise multiplication of vectors, so I've {a ^ 6 ∀ a ∈ state}
  ulong16 state6 = state2 * state4;

  // element-wise multiplication of vectors, so I've {a ^ 7 ∀ a ∈ state}
  return state6 * state;
}

// Routine for applying rescue prime hash's round key constants
// in vectorized fashion
//
// Simply adapted from
// https://github.com/itzmeanjan/ff-gpu/blob/9c57cb13e4b2d96a084da96d558fe3d4707bfcb7/rescue_prime.cpp#L65-L69
inline ulong16 apply_constants(ulong16 state, ulong16 cnst) {
  return vec_add_ff_p64(state, cnst);
}

// 64-bit Prime field modular addition of two elements
// encapsulated as vector of length 2 ( accumulating two field
// elements into single one )
//
// It's just
// https://github.com/itzmeanjan/ff-gpu/blob/7ad664cff8713b5e6bfe5527531a7532e33abd47/ff_p.cpp#L4-L22
// function re-implemented for adding two scalar elements of prime field
//
// Just the difference is two operands ( i.e. scalars ) are stored in .x and .y
// component of two-element vector
inline ulong reduce_sum_vec2(ulong2 state) {
  long over_0 = state.y >= MOD;
  ulong tmp_0 = (as_ulong(over_0) >> 63) * MOD;
  ulong b_ok = state.y - tmp_0;

  ulong tmp_1 = state.x + b_ok;
  long over_1 = state.x > (ULONG_MAX - b_ok);
  ulong tmp_2 = as_ulong(over_1) & 0x00000000ffffffff;

  ulong tmp_3 = tmp_1 + tmp_2;
  long over_2 = tmp_1 > (ULONG_MAX - tmp_2);
  ulong tmp_4 = as_ulong(over_2) & 0x00000000ffffffff;

  return tmp_3 + tmp_4;
}

// Modular accumulation of 4-prime field elements
// implemented using `reduce_sum_vec2`
//
// Given a = {0, 1, 2, 3}
//
// First it does v0 = reduce_sum_vec2(a[:2])
// then it does v1 = reduce_sum_vec2(a[2:])
//
// Finally it returns reduce_sum_vec2({v0, v1})
inline ulong reduce_sum_vec4(ulong4 state) {
  // select two elements of 4-element vector
  // and accumulate them into single element
  ulong v0 = reduce_sum_vec2(state.xy);
  // do same, just that with other elements
  ulong v1 = reduce_sum_vec2(state.zw);

  // finally create vector of length 2, with previous
  // two accumulated results and apply same function on it
  // so that four elements of vector are accumulated into single
  // field element
  return reduce_sum_vec2((ulong2)(v0, v1));
}

// Modular accumulation ( with addition operator ) of vector of 16 prime field
// elements, into single field elements, implemented using four
// `reduce_sum_vec4` function calls
//
// Given A = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
//
// Group 4 consequtive elements together, thrice, covering
// first 12 elements of state, then add three accumulated
// values and return final accmulation as 64-bit prime field
//
// Rescue prime hash function's state width is also 12
//
// v0 = reduce_sum_vec4(A[:4])
// v1 = reduce_sum_vec4(A[4:8])
// v2 = reduce_sum_vec4(A[8:12])
// v3 = 0
//
// Returns reduce_sum_vec4({v0, v1, v2, v3})
inline ulong reduce_sum(ulong16 state) {
  ulong v0 = reduce_sum_vec4(state.s0123);
  ulong v1 = reduce_sum_vec4(state.s4567);
  ulong v2 = reduce_sum_vec4(state.s89ab);

  // Note, rescue prime hash's state width is 16, so last 4
  // elements of `state` are useless, just some appended data !
  //
  // Which is exactly why I'm preparing following 4-element vector
  // with last element set to `0`
  return reduce_sum_vec4((ulong4)(v0, v1, v2, 0ul));
}

// Perform MDS matrix multiplication, updating rescue prime hash's state ( part
// of rescue permutation )
//
// I adapted
// https://github.com/itzmeanjan/ff-gpu/blob/9c57cb13e4b2d96a084da96d558fe3d4707bfcb7/rescue_prime.cpp#L52-L63
// here for vectorizing operations on hash state
inline ulong16 apply_mds(ulong16 state, ulong16 mds[STATE_WIDTH]) {
  ulong v0 = reduce_sum(vec_mul_ff_p64(state, mds[0]));
  ulong v1 = reduce_sum(vec_mul_ff_p64(state, mds[1]));
  ulong v2 = reduce_sum(vec_mul_ff_p64(state, mds[2]));
  ulong v3 = reduce_sum(vec_mul_ff_p64(state, mds[3]));

  ulong v4 = reduce_sum(vec_mul_ff_p64(state, mds[4]));
  ulong v5 = reduce_sum(vec_mul_ff_p64(state, mds[5]));
  ulong v6 = reduce_sum(vec_mul_ff_p64(state, mds[6]));
  ulong v7 = reduce_sum(vec_mul_ff_p64(state, mds[7]));

  ulong v8 = reduce_sum(vec_mul_ff_p64(state, mds[8]));
  ulong v9 = reduce_sum(vec_mul_ff_p64(state, mds[9]));
  ulong va = reduce_sum(vec_mul_ff_p64(state, mds[10]));
  ulong vb = reduce_sum(vec_mul_ff_p64(state, mds[11]));

  // filling non-rescue prime state elements with 0ul
  // because anyway they don't contribute, they're just
  // appended data to write simpler code operating on single
  // ulong16, instead of two vectors i.e. ulong8, ulong4
  return (ulong16)(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, va, vb, 0ul, 0ul,
                   0ul, 0ul);
}
