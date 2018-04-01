# C compiler
CC = g++
ICPC = icpc
CC_FLAGS = -Wall -Wextra -g -std=c++11 -qopenmp -O3 -qopt-report=5

.PHONY: clean

bfs: bfs.cpp 
	$(ICPC) $(CC_FLAGS) $^ -o $@ 

clean:
	rm -f bfs
