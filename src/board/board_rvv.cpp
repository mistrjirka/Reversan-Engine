/*
    This file is part of Reversan Engine.

    Reversan Engine is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Reversan Engine is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Reversan Engine. If not, see <https://www.gnu.org/licenses/>.
*/

// Compiller suggestion for LTO inlining
#if defined(__GNUC__) || defined(__clang__)
    #define ALWAYS_INLINE __attribute__((always_inline))
#elif defined(_MSC_VER)
    #define ALWAYS_INLINE __forceinline
#else
    #define ALWAYS_INLINE
#endif

#include "board/board.h"
#include <bit>
#include <riscv_vector.h>

// Heuristics map converted to column-interleaved byte layout for SIMD.
// Each u64 holds 8 heuristic weights (as signed bytes) for one board column,
// matching the order produced by right-shifting the bitmap and masking with 0x01.
constexpr uint64_t rvv_convert_col(int col) {
    uint64_t val = 0;
    for (int i = 0; i < 8; ++i) {
        val <<= 8;
        val |= static_cast<uint8_t>(Board::heuristics_map[i * 8 + col]);
    }
    return val;
}

static constexpr uint64_t heur_cols[8] = {
    rvv_convert_col(0), rvv_convert_col(1), rvv_convert_col(2), rvv_convert_col(3),
    rvv_convert_col(4), rvv_convert_col(5), rvv_convert_col(6), rvv_convert_col(7)
};

// Shared constants for find_moves / play_move (loaded once from .rodata)
static const uint64_t shift_vals_data[4] = {1, 7, 8, 9};
static const uint64_t col_mask_data[4] = {
    0x7e7e7e7e7e7e7e7eULL,  // SIDE_COLS_MASK
    0x7e7e7e7e7e7e7e7eULL,  // SIDE_COLS_MASK
    0xffffffffffffffffULL,  // NO_COL_MASK
    0x7e7e7e7e7e7e7e7eULL   // SIDE_COLS_MASK
};

// Helper: run the flood-fill for one color, returning valid moves bitmap.
// Assumes shift_vals_vec, col_mask_vec, and vl are already set up.
static inline uint64_t find_moves_core(
    uint64_t playing, uint64_t opponent, uint64_t free_spaces,
    vuint64m1_t shift_vals_vec, vuint64m1_t col_mask_vec, size_t vl)
{
    vuint64m1_t opponent_adjusted_vec = __riscv_vand_vx_u64m1(col_mask_vec, opponent, vl);
    vuint64m1_t ans_vec = __riscv_vmv_v_x_u64m1(playing, vl);

    vuint64m1_t left_shift_vec = __riscv_vsll_vv_u64m1(ans_vec, shift_vals_vec, vl);
    vuint64m1_t right_shift_vec = __riscv_vsrl_vv_u64m1(ans_vec, shift_vals_vec, vl);
    vuint64m1_t tmp_or_vec = __riscv_vor_vv_u64m1(left_shift_vec, right_shift_vec, vl);
    ans_vec = __riscv_vand_vv_u64m1(tmp_or_vec, opponent_adjusted_vec, vl);

    // Fully unrolled 5 iterations — eliminates branch overhead on in-order cores
    #define FLOOD_STEP \
        left_shift_vec = __riscv_vsll_vv_u64m1(ans_vec, shift_vals_vec, vl); \
        right_shift_vec = __riscv_vsrl_vv_u64m1(ans_vec, shift_vals_vec, vl); \
        tmp_or_vec = __riscv_vor_vv_u64m1(left_shift_vec, right_shift_vec, vl); \
        ans_vec = __riscv_vor_vv_u64m1(ans_vec, \
            __riscv_vand_vv_u64m1(tmp_or_vec, opponent_adjusted_vec, vl), vl);

    FLOOD_STEP
    FLOOD_STEP
    FLOOD_STEP
    FLOOD_STEP
    FLOOD_STEP
    #undef FLOOD_STEP

    left_shift_vec = __riscv_vsll_vv_u64m1(ans_vec, shift_vals_vec, vl);
    right_shift_vec = __riscv_vsrl_vv_u64m1(ans_vec, shift_vals_vec, vl);
    tmp_or_vec = __riscv_vor_vv_u64m1(left_shift_vec, right_shift_vec, vl);

    vuint64m1_t zero_scalar = __riscv_vmv_v_x_u64m1(0, 1);
    vuint64m1_t reduced = __riscv_vredor_vs_u64m1_u64m1(tmp_or_vec, zero_scalar, vl);
    return __riscv_vmv_x_s_u64m1_u64(reduced) & free_spaces;
}

ALWAYS_INLINE int Board::rate_board() const {
    // --- Heuristic scoring (vectorized) ---
    const size_t vl8 = __riscv_vsetvl_e64m2(8);
    static const uint64_t shift_vals_8[8] = {7, 6, 5, 4, 3, 2, 1, 0};

    vuint64m2_t shift_vec = __riscv_vle64_v_u64m2(shift_vals_8, vl8);

    vuint64m2_t white_vec = __riscv_vmv_v_x_u64m2(white_bitmap, vl8);
    white_vec = __riscv_vsrl_vv_u64m2(white_vec, shift_vec, vl8);
    white_vec = __riscv_vand_vx_u64m2(white_vec, 0x0101010101010101ULL, vl8);

    vuint64m2_t black_vec = __riscv_vmv_v_x_u64m2(black_bitmap, vl8);
    black_vec = __riscv_vsrl_vv_u64m2(black_vec, shift_vec, vl8);
    black_vec = __riscv_vand_vx_u64m2(black_vec, 0x0101010101010101ULL, vl8);

    vuint8m2_t white_bytes = __riscv_vreinterpret_v_u64m2_u8m2(white_vec);
    vuint8m2_t black_bytes = __riscv_vreinterpret_v_u64m2_u8m2(black_vec);

    const size_t vl64 = __riscv_vsetvl_e8m2(64);

    vint8m2_t white_i8 = __riscv_vreinterpret_v_u8m2_i8m2(white_bytes);
    vint8m2_t black_i8 = __riscv_vreinterpret_v_u8m2_i8m2(black_bytes);
    vint8m2_t delta = __riscv_vsub_vv_i8m2(white_i8, black_i8, vl64);

    vint8m2_t heur_vec = __riscv_vle8_v_i8m2((const int8_t*)heur_cols, vl64);
    vint16m4_t products = __riscv_vwmul_vv_i16m4(delta, heur_vec, vl64);

    vint16m1_t zero_i16 = __riscv_vmv_v_x_i16m1(0, 1);
    vint16m1_t sum_vec = __riscv_vredsum_vs_i16m4_i16m1(products, zero_i16, vl64);
    int score = (int)__riscv_vmv_x_s_i16m1_i16(sum_vec);

    // --- Mobility scoring: inline both find_moves calls, sharing setup ---
    uint64_t free_spaces = ~(white_bitmap | black_bitmap);
    const size_t vl4 = __riscv_vsetvl_e64m1(4);
    vuint64m1_t sv = __riscv_vle64_v_u64m1(shift_vals_data, vl4);
    vuint64m1_t cm = __riscv_vle64_v_u64m1(col_mask_data, vl4);

    uint64_t white_moves = find_moves_core(white_bitmap, black_bitmap, free_spaces, sv, cm, vl4);
    uint64_t black_moves = find_moves_core(black_bitmap, white_bitmap, free_spaces, sv, cm, vl4);

    int moves_delta = std::popcount(white_moves) - std::popcount(black_moves);
    score += 10 * moves_delta;

    return score;
}

ALWAYS_INLINE uint64_t Board::find_moves(bool color) const {
    uint64_t free_spaces = ~(white_bitmap | black_bitmap);

    uint64_t playing, opponent;
    if (color) {
        playing = white_bitmap;
        opponent = black_bitmap;
    } else {
        playing = black_bitmap;
        opponent = white_bitmap;
    }

    const size_t vl = __riscv_vsetvl_e64m1(4);
    vuint64m1_t sv = __riscv_vle64_v_u64m1(shift_vals_data, vl);
    vuint64m1_t cm = __riscv_vle64_v_u64m1(col_mask_data, vl);

    return find_moves_core(playing, opponent, free_spaces, sv, cm, vl);
}

ALWAYS_INLINE void Board::play_move(bool color, uint64_t move) {
    uint64_t playing;
    uint64_t opponent;
    if (color) {
        playing = white_bitmap;
        opponent = black_bitmap;
    }
    else {
        playing = black_bitmap;
        opponent = white_bitmap;
    }

    const size_t vl = __riscv_vsetvl_e64m1(4);
    vuint64m1_t shift_vals_vec = __riscv_vle64_v_u64m1(shift_vals_data, vl);
    vuint64m1_t col_mask_vec = __riscv_vle64_v_u64m1(col_mask_data, vl);
    vuint64m1_t playing_vec = __riscv_vmv_v_x_u64m1(playing, vl);
    vuint64m1_t opponent_adjusted_vec = __riscv_vand_vx_u64m1(col_mask_vec, opponent, vl);

    vuint64m1_t move_vec = __riscv_vmv_v_x_u64m1(move, vl);
    vuint64m1_t left_shift_vec = __riscv_vsll_vv_u64m1(move_vec, shift_vals_vec, vl);
    vuint64m1_t right_shift_vec = __riscv_vsrl_vv_u64m1(move_vec, shift_vals_vec, vl);
    left_shift_vec = __riscv_vand_vv_u64m1(left_shift_vec, opponent_adjusted_vec, vl);
    right_shift_vec = __riscv_vand_vv_u64m1(right_shift_vec, opponent_adjusted_vec, vl);

    vuint64m1_t left_shift_vec_next;
    vuint64m1_t right_shift_vec_next;

    // Fully unrolled 6 iterations — eliminates branch overhead on in-order X60
    #define PLAY_STEP \
        left_shift_vec_next = __riscv_vsll_vv_u64m1(left_shift_vec, shift_vals_vec, vl); \
        right_shift_vec_next = __riscv_vsrl_vv_u64m1(right_shift_vec, shift_vals_vec, vl); \
        left_shift_vec = __riscv_vand_vv_u64m1( \
            __riscv_vor_vv_u64m1(left_shift_vec, left_shift_vec_next, vl), \
            opponent_adjusted_vec, vl); \
        right_shift_vec = __riscv_vand_vv_u64m1( \
            __riscv_vor_vv_u64m1(right_shift_vec, right_shift_vec_next, vl), \
            opponent_adjusted_vec, vl);

    PLAY_STEP
    PLAY_STEP
    PLAY_STEP
    PLAY_STEP
    PLAY_STEP
    PLAY_STEP
    #undef PLAY_STEP

    // Branch-free capture validation
    vuint64m1_t zero_vec = __riscv_vmv_v_x_u64m1(0, vl);
    vuint64m1_t left_check = __riscv_vand_vv_u64m1(left_shift_vec_next, playing_vec, vl);
    vuint64m1_t right_check = __riscv_vand_vv_u64m1(right_shift_vec_next, playing_vec, vl);

    vbool64_t left_valid = __riscv_vmsne_vv_u64m1_b64(left_check, zero_vec, vl);
    vbool64_t right_valid = __riscv_vmsne_vv_u64m1_b64(right_check, zero_vec, vl);

    vuint64m1_t capture_left = __riscv_vmerge_vvm_u64m1(zero_vec, left_shift_vec, left_valid, vl);
    vuint64m1_t capture_right = __riscv_vmerge_vvm_u64m1(zero_vec, right_shift_vec, right_valid, vl);

    vuint64m1_t capture_all = __riscv_vor_vv_u64m1(capture_left, capture_right, vl);
    vuint64m1_t zero_scalar = __riscv_vmv_v_x_u64m1(0, 1);
    vuint64m1_t reduced = __riscv_vredor_vs_u64m1_u64m1(capture_all, zero_scalar, vl);
    uint64_t capture = __riscv_vmv_x_s_u64m1_u64(reduced);

    playing |= move;
    playing |= capture;
    opponent ^= capture;

    if (color) {
        white_bitmap = playing;
        black_bitmap = opponent;
    }
    else {
        white_bitmap = opponent;
        black_bitmap = playing;
    }
}