CXX = g++
CXXFLAGS = -std=c++17 -pthread -Wall -Wextra -Iinclude

SRC_FILES = src/engines.cpp src/simulator.cpp
TEST_FILE = tests/test_simulator.cpp

TARGET = test_simulator

all: $(TARGET)

$(TARGET): $(SRC_FILES) $(TEST_FILE)
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f $(TARGET)
