# C compiler
ICPC = icpc
#CC_FLAGS = -Wall -Wextra -g -std=c++11 -qopenmp -O3 -qopt-report=5 -qopt-report-phase=vec
CC_FLAGS = -Wall -Wextra -g -std=c++11 -qopenmp -O3 -qopt-prefetch=5 -qopt-report=5 -qopt-report-phase=vec
#CC_FLAGS = -Wall -Wextra -g -std=c++11 -qopenmp

.PHONY: clean

bfs: bfs.cpp 
	$(ICPC) $(CC_FLAGS) $^ -o $@ 

clean:
	rm -f bfs
