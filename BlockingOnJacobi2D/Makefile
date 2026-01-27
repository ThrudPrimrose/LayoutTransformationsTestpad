CXX = g++
CXXFLAGS = -O3 -fopenmp -march=native -ffast-math -Wall
LDFLAGS = -fopenmp

TARGETS = jacobi2d_baseline.out jacobi2d_tiled.out jacobi2d_boundary.out jacobi2d_blocked_tiled.out

all: $(TARGETS)

jacobi2d_baseline.out: jacobi2d_baseline.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

jacobi2d_tiled.out: jacobi2d_tiled.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

jacobi2d_boundary.out: jacobi2d_boundary.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)


jacobi2d_blocked_tiled.out: jacobi2d_blocked_tiled.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGETS) *.bin *.o

.PHONY: all clean
