# C compiler
CC = g++
ICPC = icpc
#ICPC = icc
#CC_FLAGS = -Wall -Wextra -g -qopenmp -O3 -qopt-report=5 -vec-threshold0
CC_FLAGS = -Wall -Wextra -g -qopenmp -O3 -qopt-report=5
#CC_FLAGS = -Wall -Wextra -g -fopenmp -O3 -qopt-report=5

.PHONY: clean all
#all: bfs bfs_offload
#all: bfs bfs_auto

bfs: bfs.cpp 
	$(ICPC) $(CC_FLAGS) $^ -o $@ 

#bfs_auto: bfs.cpp
#	$(ICPC) $(CC_FLAGS_AUTO) $^ -o $@ 

#bfs_offload: bfs.cpp
#	$(ICC) $(CC_FLAGS) -DOMP_OFFLOAD bfs.cpp -o bfs_offload

clean:
#rm -f bfs bfs_offload result.txt
	rm -f bfs bfs_auto
