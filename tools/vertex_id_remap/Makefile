# C compiler
ICPC = icpc

ifeq (${debug}, 1)
	CPPFLAGS = -w2 -g -std=c++14 -qopenmp -DONEDEBUG -O0
else
	CPPFLAGS = -w2 -g -std=c++14 -qopenmp -O3 -qopt-report=5 -qopt-report-phase=vec -inline-level=2 -inline-forceinline
endif

ifeq (${weighted}, 1)
	CPPFLAGS += -DWEIGHTED
endif

.PHONY: clean, all

all: bfs

bfs: bfs.cpp 
	$(ICPC) $(CPPFLAGS) $^ -o $@ 

clean:
	rm -f bfs
