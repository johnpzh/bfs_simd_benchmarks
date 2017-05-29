# C compiler
ICPC = icpc
#CC_FLAGS = -Wall -Wextra -g -qopenmp -O3 -qopt-report=5 -vec-threshold0
CC_FLAGS = -Wall -Wextra -g -qopenmp -O3 -qopt-report=5
#CC_FLAGS = -Wall -Wextra -g 

.PHONY: clean all

bfs: bfs.cpp 
	$(ICPC) $(CC_FLAGS) $^ -o $@ 

clean:
	rm -f bfs
