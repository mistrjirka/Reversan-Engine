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

ALWAYS_INLINE int Board::rate_board() const {
    // --- Phase 1: Extract per-column bit presence as bytes (e64, 8 elements) ---
    // Shift bitmap by {7,6,...,0} to align each column's bits into bit 0 of each byte,
    // then mask with 0x01. Result: 8 × u64 = 64 bytes, each 0 or 1.
    // Requires LMUL=2 to hold 8 × u64 on VLEN=256.
    const size_t vl8 = __riscv_vsetvl_e64m2(8);
    static const uint64_t shift_vals[8] = {7, 6, 5, 4, 3, 2, 1, 0};

    vuint64m2_t shift_vec = __riscv_vle64_v_u64m2(shift_vals, vl8);

    vuint64m2_t white_vec = __riscv_vmv_v_x_u64m2(white_bitmap, vl8);
    white_vec = __riscv_vsrl_vv_u64m2(white_vec, shift_vec, vl8);
    white_vec = __riscv_vand_vx_u64m2(white_vec, 0x0101010101010101ULL, vl8);

    vuint64m2_t black_vec = __riscv_vmv_v_x_u64m2(black_bitmap, vl8);
    black_vec = __riscv_vsrl_vv_u64m2(black_vec, shift_vec, vl8);
    black_vec = __riscv_vand_vx_u64m2(black_vec, 0x0101010101010101ULL, vl8);

    // --- Phase 2: Byte-level multiply + reduce (e8/e16, 64 elements) ---
    // Reinterpret the 8×u64 as 64×u8, then compute (white - black) per position,
    // widening-multiply by heuristic weights, and reduce-sum.
    vuint8m2_t white_bytes = __riscv_vreinterpret_v_u64m2_u8m2(white_vec);
    vuint8m2_t black_bytes = __riscv_vreinterpret_v_u64m2_u8m2(black_vec);

    const size_t vl64 = __riscv_vsetvl_e8m2(64);

    // delta = white - black ∈ {-1, 0, 1} per position
    vint8m2_t white_i8 = __riscv_vreinterpret_v_u8m2_i8m2(white_bytes);
    vint8m2_t black_i8 = __riscv_vreinterpret_v_u8m2_i8m2(black_bytes);
    vint8m2_t delta = __riscv_vsub_vv_i8m2(white_i8, black_i8, vl64);

    // Load column-layout heuristic weights as 64 × i8
    vint8m2_t heur_vec = __riscv_vle8_v_i8m2((const int8_t*)heur_cols, vl64);

    // Widening signed multiply: i8 × i8 → i16 (m2 × m2 → m4)
    vint16m4_t products = __riscv_vwmul_vv_i16m4(delta, heur_vec, vl64);

    // Reduce sum all 64 × i16 → single i16
    vint16m1_t zero_i16 = __riscv_vmv_v_x_i16m1(0, 1);
    vint16m1_t sum_vec = __riscv_vredsum_vs_i16m4_i16m1(products, zero_i16, vl64);
    int score = (int)__riscv_vmv_x_s_i16m1_i16(sum_vec);

    // --- Mobility scoring ---
    int moves_delta = std::popcount(find_moves(true)) - std::popcount(find_moves(false));
    score += 10 * moves_delta;

    return score;
}

ALWAYS_INLINE uint64_t Board::find_moves(bool color) const {
    uint64_t free_spaces = ~(white_bitmap | black_bitmap);

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

    // On VLEN=256 (SpacemiT X60), m1 holds 4 × u64 — all 4 directions fit
    const size_t vl = __riscv_vsetvl_e64m1(4);

    // Static arrays stay in .rodata, always hot in L1 — avoids
    // stack materialization overhead on every call
    static const uint64_t shift_vals_data[4] = {1, 7, 8, 9};
    static const uint64_t col_mask_data[4] = {
        Masks::SIDE_COLS_MASK,
        Masks::SIDE_COLS_MASK,
        Masks::NO_COL_MASK,
        Masks::SIDE_COLS_MASK
    };

    vuint64m1_t shift_vals_vec = __riscv_vle64_v_u64m1(shift_vals_data, vl);
    vuint64m1_t col_mask_vec = __riscv_vle64_v_u64m1(col_mask_data, vl);
    vuint64m1_t opponent_adjusted_vec = __riscv_vand_vx_u64m1(col_mask_vec, opponent, vl);
    vuint64m1_t ans_vec = __riscv_vmv_v_x_u64m1(playing, vl);

    vuint64m1_t left_shift_vec = __riscv_vsll_vv_u64m1(ans_vec, shift_vals_vec, vl);
    vuint64m1_t right_shift_vec = __riscv_vsrl_vv_u64m1(ans_vec, shift_vals_vec, vl);
    vuint64m1_t tmp_or_vec = __riscv_vor_vv_u64m1(left_shift_vec, right_shift_vec, vl);
    ans_vec = __riscv_vand_vv_u64m1(tmp_or_vec, opponent_adjusted_vec, vl);

    for (int i = 0; i < 5; ++i) {
        left_shift_vec = __riscv_vsll_vv_u64m1(ans_vec, shift_vals_vec, vl);
        right_shift_vec = __riscv_vsrl_vv_u64m1(ans_vec, shift_vals_vec, vl);
        tmp_or_vec = __riscv_vor_vv_u64m1(left_shift_vec, right_shift_vec, vl);
        vuint64m1_t tmp_and_vec = __riscv_vand_vv_u64m1(tmp_or_vec, opponent_adjusted_vec, vl);
        ans_vec = __riscv_vor_vv_u64m1(ans_vec, tmp_and_vec, vl);
    }

    left_shift_vec = __riscv_vsll_vv_u64m1(ans_vec, shift_vals_vec, vl);
    right_shift_vec = __riscv_vsrl_vv_u64m1(ans_vec, shift_vals_vec, vl);
    tmp_or_vec = __riscv_vor_vv_u64m1(left_shift_vec, right_shift_vec, vl);

    // Vector reduction OR — merge all 4 directions in-register, no store+load round-trip
    vuint64m1_t zero_scalar = __riscv_vmv_v_x_u64m1(0, 1);
    vuint64m1_t reduced = __riscv_vredor_vs_u64m1_u64m1(tmp_or_vec, zero_scalar, vl);
    uint64_t valid_moves = __riscv_vmv_x_s_u64m1_u64(reduced);

    valid_moves &= free_spaces;
    return valid_moves;
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
    const uint64_t shift_vals_data[4] = {1, 7, 8, 9};
    const uint64_t col_mask_data[4] = {
        Masks::SIDE_COLS_MASK,
        Masks::SIDE_COLS_MASK,
        Masks::NO_COL_MASK,
        Masks::SIDE_COLS_MASK
    };

    vuint64m1_t shift_vals_vec = __riscv_vle64_v_u64m1(shift_vals_data, vl);
    vuint64m1_t col_mask_vec = __riscv_vle64_v_u64m1(col_mask_data, vl);
    vuint64m1_t move_vec = __riscv_vmv_v_x_u64m1(move, vl);
    vuint64m1_t opponent_vec = __riscv_vmv_v_x_u64m1(opponent, vl);
    vuint64m1_t opponent_adjusted_vec = __riscv_vand_vv_u64m1(opponent_vec, col_mask_vec, vl);

    vuint64m1_t left_shift_vec = __riscv_vsll_vv_u64m1(move_vec, shift_vals_vec, vl);
    vuint64m1_t right_shift_vec = __riscv_vsrl_vv_u64m1(move_vec, shift_vals_vec, vl);
    left_shift_vec = __riscv_vand_vv_u64m1(left_shift_vec, opponent_adjusted_vec, vl);
    right_shift_vec = __riscv_vand_vv_u64m1(right_shift_vec, opponent_adjusted_vec, vl);

    vuint64m1_t left_shift_vec_next = left_shift_vec;
    vuint64m1_t right_shift_vec_next = right_shift_vec;

    for (int i = 0; i < 6; ++i) {
        left_shift_vec_next = __riscv_vsll_vv_u64m1(left_shift_vec, shift_vals_vec, vl);
        right_shift_vec_next = __riscv_vsrl_vv_u64m1(right_shift_vec, shift_vals_vec, vl);

        left_shift_vec = __riscv_vor_vv_u64m1(left_shift_vec, left_shift_vec_next, vl);
        right_shift_vec = __riscv_vor_vv_u64m1(right_shift_vec, right_shift_vec_next, vl);

        left_shift_vec = __riscv_vand_vv_u64m1(left_shift_vec, opponent_adjusted_vec, vl);
        right_shift_vec = __riscv_vand_vv_u64m1(right_shift_vec, opponent_adjusted_vec, vl);
    }

    uint64_t left_data[4];
    uint64_t right_data[4];
    uint64_t left_next_data[4];
    uint64_t right_next_data[4];
    __riscv_vse64_v_u64m1(left_data, left_shift_vec, vl);
    __riscv_vse64_v_u64m1(right_data, right_shift_vec, vl);
    __riscv_vse64_v_u64m1(left_next_data, left_shift_vec_next, vl);
    __riscv_vse64_v_u64m1(right_next_data, right_shift_vec_next, vl);

    uint64_t capture = 0;
    for (int i = 0; i < 4; ++i) {
        if ((left_next_data[i] & playing) != 0) {
            capture |= left_data[i];
        }
        if ((right_next_data[i] & playing) != 0) {
            capture |= right_data[i];
        }
    }

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