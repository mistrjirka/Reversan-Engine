#       This file is part of Reversan Engine.
#
#       Reversan Engine is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   Reversan Engine is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with Reversan Engine. If not, see <https://www.gnu.org/licenses/>. 

# Select compiller of your choice, tested only with g++
CXX = g++
LINKER = g++

# Add include path
CXX_FLAGS = -Iinclude

# Select compile flags
CXX_FLAGS += -std=c++20 -O3 -flto -Wall -Wno-attributes
LINKER_FLAGS = -flto

# Add source and build path
SOURCE_DIR = src
BUILD_DIR = build
TARGET_EXE = reversan

# --- Sources Setup ---
SOURCES  = main.cpp
SOURCES += app/app.cpp
SOURCES += board/board_state.cpp
SOURCES += engine/alphabeta.cpp
SOURCES += engine/move_order.cpp
SOURCES += engine/negascout.cpp
SOURCES += engine/transposition_table.cpp
SOURCES += ui/terminal.cpp
SOURCES += utils/parser.cpp
SOURCES += utils/thread_manager.cpp
OBJECTS = $(addprefix $(BUILD_DIR)/,$(SOURCES:%.cpp=%.o))

# Sources when building without any explicit SIMD instructions
SOURCES_NOSIMD = board/board_nosimd.cpp
OBJECTS_NOSIMD = $(OBJECTS) $(addprefix $(BUILD_DIR)/,$(SOURCES_NOSIMD:%.cpp=%.o))

# Sources when building with explicit AVX2 instructions
SOURCES_AVX2 = board/board_avx2.cpp
OBJECTS_AVX2 = $(OBJECTS) $(addprefix $(BUILD_DIR)/,$(SOURCES_AVX2:%.cpp=%.o))

# Sources when building with explicit RVV 1.0 instructions
SOURCES_RVV = board/board_rvv.cpp
OBJECTS_RVV = $(OBJECTS) $(addprefix $(BUILD_DIR)/,$(SOURCES_RVV:%.cpp=%.o))

# --- Helper: Detect System Architecture ---
HOST_ARCH := $(shell uname -m)

# -------------------------------------------------------------------------
# TARGETS
# -------------------------------------------------------------------------

.PHONY: all debug clean K1_X60 auto auto_rvv auto_x64

# Default target (x86 AVX2)
all: CXX_FLAGS += -mavx2
all: $(OBJECTS_AVX2)
	$(LINKER) $(LINKER_FLAGS) $^ -o $(TARGET_EXE)

debug: CXX_FLAGS += -pg
debug: LINKER_FLAGS += -pg
debug: all

no_simd: $(OBJECTS_NOSIMD)
	$(LINKER) $(LINKER_FLAGS) $^ -o $(TARGET_EXE)

# Your original handwritten RVV target
rvv: CXX_FLAGS := $(filter-out -flto,$(CXX_FLAGS))
rvv: LINKER_FLAGS := $(filter-out -flto,$(LINKER_FLAGS))
rvv: CXX_FLAGS += -march=rv64gcv -mabi=lp64d
rvv: LINKER_FLAGS += -march=rv64gcv -mabi=lp64d -static
rvv: $(OBJECTS_RVV)
	$(LINKER) $(LINKER_FLAGS) $^ -o $(TARGET_EXE)

# --- 1. Specific Target: SpacemiT K1 / X60 (Your Orange Pi) ---
# Manual definition is safest for this board.
K1_ARCH_STR = -march=rv64gcv_zba_zbb_zbc_zbs_zkt_zfh_zfhmin_zvfh_zvfhmin_zicond_zicbom_zicbop_zicboz_zvl256b -mabi=lp64d

K1_X60: CXX_FLAGS := $(filter-out -flto,$(CXX_FLAGS))
K1_X60: LINKER_FLAGS := $(filter-out -flto,$(LINKER_FLAGS))
K1_X60: CXX_FLAGS += $(K1_ARCH_STR) -ftree-vectorize
K1_X60: LINKER_FLAGS += $(K1_ARCH_STR) -static
K1_X60: $(OBJECTS_NOSIMD)
	@echo "[INFO] Building specific target for SpacemiT K1 (X60)..."
	$(LINKER) $(LINKER_FLAGS) $^ -o $(TARGET_EXE)

# --- 2. Auto Target: RISC-V (Smart Detection) ---
# Parses /proc/cpuinfo. We use sed to remove 's' extensions (like sstc, svpbmt) which break GCC.
DETECTED_ISA := $(shell grep "^isa" /proc/cpuinfo 2>/dev/null | head -n 1 | cut -d: -f2 | sed 's/ //g')
SAFE_ISA := $(shell echo $(DETECTED_ISA) | sed 's/_s[a-z]*//g' | sed 's/__/_/g' | sed 's/_zicntr//g' | sed 's/_zihpm//g')

auto_rvv: CXX_FLAGS := $(filter-out -flto,$(CXX_FLAGS))
auto_rvv: LINKER_FLAGS := $(filter-out -flto,$(LINKER_FLAGS))
auto_rvv: CXX_FLAGS += -march=$(SAFE_ISA) -mabi=lp64d -ftree-vectorize
auto_rvv: LINKER_FLAGS += -march=$(SAFE_ISA) -mabi=lp64d -static
auto_rvv: $(OBJECTS_NOSIMD)
	@echo "[INFO] Auto-detected RISC-V Architecture: $(SAFE_ISA)"
	$(LINKER) $(LINKER_FLAGS) $^ -o $(TARGET_EXE)

# --- 3. Auto Target: x86_64 (Intel/AMD) ---
auto_x64: CXX_FLAGS += -march=native -ftree-vectorize
auto_x64: $(OBJECTS_NOSIMD)
	@echo "[INFO] Auto-detected x64 Architecture"
	$(LINKER) $(LINKER_FLAGS) $^ -o $(TARGET_EXE)

# --- 4. Master 'auto' Target ---
auto:
ifeq ($(HOST_ARCH),riscv64)
	$(MAKE) auto_rvv
else ifeq ($(HOST_ARCH),x86_64)
	$(MAKE) auto_x64
else
	@echo "[ERROR] Unknown architecture: $(HOST_ARCH). Please use a specific target."
endif

debug_no_simd: CXX_FLAGS += -pg
debug_no_simd: LINKER_FLAGS += -pg
debug_no_simd: no_simd

debug_rvv: CXX_FLAGS += -pg
debug_rvv: LINKER_FLAGS += -pg
debug_rvv: rvv

clean:
	rm -f -r $(BUILD_DIR)
	rm -f $(TARGET_EXE)

$(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir $@
