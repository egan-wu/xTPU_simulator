CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -Iinclude
BUILD_DIR = build

# Source files
SRCS = src/main.cpp \
       src/control_unit.cpp \
       src/engines/sdma_engine.cpp \
       src/engines/idma_engine.cpp \
       src/engines/mxu_engine.cpp \
       src/engines/vector_engine.cpp \
       src/engines/processing_unit.cpp

# Object files (mapped to build directory)
OBJS = $(SRCS:%.cpp=$(BUILD_DIR)/%.o)

# Target executable
TARGET = tpu_simulator

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Rule for object files
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean
