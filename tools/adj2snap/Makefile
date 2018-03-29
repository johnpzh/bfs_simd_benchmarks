# C compiler
ICPC = icpc

ifeq (${debug}, 1)
	CC_FLAGS = -Wall -Wextra -g -std=c++14 -qopenmp -DONEDEBUG
else
	CC_FLAGS = -Wall -Wextra -g -std=c++14 -qopenmp -O3 -qopt-report=5 -qopt-report-phase=vec -inline-level=2 -inline-forceinline
endif

.PHONY: clean, all

all: adj2snap

adj2snap: adj2snap.cpp 
	$(ICPC) $(CC_FLAGS) $^ -o $@ 

clean:
	rm -f adj2snap
