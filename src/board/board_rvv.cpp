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

ALWAYS_INLINE int Board::rate_board() const {
    int score = 0;
    for (int i = 0; i < 64; ++i) {
        score += ((white_bitmap >> i) & 1) * heuristics_map[63 - i];
        score -= ((black_bitmap >> i) & 1) * heuristics_map[63 - i];
    }

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