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

#include "app/app.h"
#include "ui/terminal.h"
#include "engine/negascout.h"
#include "engine/alphabeta.h"
#include "utils/parser.h"
#include "board/board.h"
#include <signal.h>
#include <chrono>
#include <iostream>
#include <cstring>

// needs to be file-global to be accessible in sig function
static UI *ui = nullptr;
static Engine *engine = nullptr;

// restores terminal state even after ctrl-c or other failure
void handle_sig(int sig) {
    // safely dealocate resources
    if (ui) delete ui;
    if (engine) delete engine;
    exit(sig);
}

static void run_profile() {
    using clock = std::chrono::high_resolution_clock;
    constexpr int ITERS = 10'000'000;

    // Use multiple board states to avoid branch-predictor / cache cheating
    Board boards[3] = {
        Board::States::INITIAL,
        Board::States::TEST,
        Board::States::BENCHMARK
    };

    volatile uint64_t sink = 0; // prevent dead-code elimination
    volatile int isink = 0;

    // --- find_moves ---
    {
        auto t0 = clock::now();
        for (int i = 0; i < ITERS; ++i) {
            sink = boards[i % 3].find_moves(i & 1);
        }
        auto t1 = clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double ns_per = ms * 1e6 / ITERS;
        std::cout << "find_moves : " << ms << " ms total, "
                  << ns_per << " ns/call  (" << ITERS << " iters)\n";
    }

    // --- rate_board ---
    {
        auto t0 = clock::now();
        for (int i = 0; i < ITERS; ++i) {
            isink = boards[i % 3].rate_board();
        }
        auto t1 = clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double ns_per = ms * 1e6 / ITERS;
        std::cout << "rate_board : " << ms << " ms total, "
                  << ns_per << " ns/call  (" << ITERS << " iters)\n";
    }

    // --- play_move ---
    {
        auto t0 = clock::now();
        for (int i = 0; i < ITERS; ++i) {
            Board b = boards[i % 3];
            uint64_t moves = b.find_moves(i & 1);
            if (moves) {
                // pick lowest set bit as the move
                uint64_t move = moves & (-moves);
                b.play_move(i & 1, move);
            }
            sink = b.white() ^ b.black();
        }
        auto t1 = clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double ns_per = ms * 1e6 / ITERS;
        std::cout << "play_move  : " << ms << " ms total, "
                  << ns_per << " ns/call  (" << ITERS << " iters, includes find_moves)\n";
    }

    (void)sink;
    (void)isink;
}

int main(int argc, char **argv) {
    // Check for --profile before normal parsing
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--profile") == 0) {
            run_profile();
            return 0;
        }
    }

    // prepare signal handler
    signal(SIGINT, handle_sig);

    // parse arguments
    Parser parser;
    if (!parser.parse(argc, argv)) return 1;

    // initialize engine
    if (parser.get_alg() == Engine::Alg::ALPHABETA) {
        engine = new Alphabeta(parser.get_settings());
    }
    else if (parser.get_alg() == Engine::Alg::NEGASCOUT && parser.get_settings().thread_count > 1) {
        engine = new NegascoutParallel(parser.get_settings());
    }
    else {
        engine = new Negascout(parser.get_settings());
    }

    // initialize terminal
    ui = new Terminal(parser.get_style());

    // initialize app
    App app(parser.get_mode(), ui, engine);
    app.run();

    // dealocate resources and exit
    delete ui;
    delete engine;
    return 0;
}
