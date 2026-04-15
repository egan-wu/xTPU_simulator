CXX      = g++
# CR3-5: -MMD 自動產生 .d 依賴檔，-MP 為每個 header 產生 phony target
#         避免 header 被刪除後 make 因找不到依賴而失敗
DEPFLAGS = -MMD -MP

# P3-3: LPDDR5-sim submodule 路徑
LPDDR5_DIR     = submodule/lpddr5-sim
LPDDR5_INCLUDE = -I$(LPDDR5_DIR)/include
LPDDR5_LIB     = $(LPDDR5_DIR)/build/liblpddr5_sim.a

CXXFLAGS_BASE = -std=c++17 -pthread -Wall -Wextra -Iinclude $(LPDDR5_INCLUDE)
CXXFLAGS      = $(CXXFLAGS_BASE) -O2

SRC_FILES  = src/engines.cpp src/simulator.cpp src/lpddr5_adapter.cpp
TEST_FILE  = tests/test_simulator.cpp
ALL_SRCS   = $(SRC_FILES) $(TEST_FILE)

# 物件檔與依賴檔放在與 .cpp 相同目錄下
OBJS = $(ALL_SRCS:.cpp=.o)
DEPS = $(ALL_SRCS:.cpp=.d)

TARGET = test_simulator

# P3-CR-5: 自動建置 lpddr5-sim 靜態庫（P4-3）
# 只要 liblpddr5_sim.a 不存在或比 submodule 原始碼舊，就重新 build
$(LPDDR5_LIB):
	@echo "[make] Building lpddr5-sim submodule..."
	cmake -S $(LPDDR5_DIR) -B $(LPDDR5_DIR)/build \
	    -DCMAKE_BUILD_TYPE=Release \
	    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
	    -DBUILD_TESTING=OFF 2>&1 | tail -5
	cmake --build $(LPDDR5_DIR)/build --parallel
	@echo "[make] lpddr5-sim built: $(LPDDR5_LIB)"

.PHONY: all clean debug release test tsan asan lpddr5

# ── 預設 target ───────────────────────────────────────────────────────────────
all: $(TARGET)

$(TARGET): $(OBJS) $(LPDDR5_LIB)
	$(CXX) $(CXXFLAGS) $(OBJS) $(LPDDR5_LIB) -o $@

# ── 通用規則：.cpp → .o（同時產生 .d 依賴檔）────────────────────────────────
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

# 引入自動產生的依賴檔（第一次 build 時 .d 不存在，- 前綴忽略錯誤）
-include $(DEPS)

# ── P4-3: 便利 targets ────────────────────────────────────────────────────────

# debug：無最佳化、完整 debug 符號
debug: CXXFLAGS = $(CXXFLAGS_BASE) -O0 -g3 -DDEBUG
debug: $(TARGET)

# release：最高最佳化、strip debug info
release: CXXFLAGS = $(CXXFLAGS_BASE) -O3 -DNDEBUG
release: $(TARGET)

# test：build 並執行測試（回傳 test runner 的 exit code）
test: all
	./$(TARGET)

# lpddr5：強制重新 build lpddr5-sim submodule（P3-CR-5）
lpddr5:
	rm -f $(LPDDR5_LIB)
	$(MAKE) $(LPDDR5_LIB)

# tsan：ThreadSanitizer build（檢測 data race）
tsan: CXXFLAGS = $(CXXFLAGS_BASE) -O1 -g -fsanitize=thread
tsan: clean $(TARGET)
	./$(TARGET)

# asan：AddressSanitizer build（檢測 buffer overflow、use-after-free 等）
asan: CXXFLAGS = $(CXXFLAGS_BASE) -O1 -g -fsanitize=address,undefined
asan: clean $(TARGET)
	./$(TARGET)

# ── P5-8: xbin_runner ────────────────────────────────────────────────────────
XBIN_RUNNER_SRC = tools/xbin_runner.cpp
XBIN_RUNNER_OBJ = $(XBIN_RUNNER_SRC:.cpp=.o)
XBIN_RUNNER     = xbin_runner

$(XBIN_RUNNER): $(XBIN_RUNNER_OBJ) src/engines.o src/simulator.o src/lpddr5_adapter.o $(LPDDR5_LIB)
	$(CXX) $(CXXFLAGS) $^ -o $@

# ── clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -f $(TARGET) $(XBIN_RUNNER) $(OBJS) $(XBIN_RUNNER_OBJ) $(DEPS) tools/*.d
