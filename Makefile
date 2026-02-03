CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -Iinclude

# Source files
SRCS = src/main.cpp \
       src/simulator.cpp \
       src/engines/sdma_engine.cpp \
       src/engines/idma_engine.cpp \
       src/engines/mxu_engine.cpp \
       src/engines/vector_engine.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Target executable
TARGET = tpu_simulator

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
