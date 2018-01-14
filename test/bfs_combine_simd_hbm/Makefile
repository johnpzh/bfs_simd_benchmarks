# C compiler
ICPC = icpc

ifeq (${debug}, 1)
	CC_FLAGS = -Wall -Wextra -g -std=c++14 -qopenmp -DONEDEBUG
else
	CC_FLAGS = -Wall -Wextra -g -std=c++14 -qopenmp -O3 -xMIC-AVX512 -qopt-report=5 -qopt-report-phase=vec -inline-level=2 -inline-forceinline -lmemkind
endif

.PHONY: clean, all

all: bfs

bfs: bfs.cpp 
	$(ICPC) $(CC_FLAGS) $^ -o $@ 

clean:
	rm -f bfs
