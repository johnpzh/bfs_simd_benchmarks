ICPC = icpc

ifeq (${debug}, 1)
	CPPFLAGS = -Wall -Wextra -g -std=c++14 -qopenmp -DONEDEBUG
else
	CPPFLAGS = -Wall -Wextra -g -std=c++14 -qopenmp -O3 -xMIC-AVX512 -qopt-report=5 -qopt-report-phase=vec -inline-level=2 -inline-forceinline
endif

.PHONY: clean

all: sssp

sssp: sssp.cpp
	$(ICPC) $(CPPFLAGS) -o $@ $^

clean:
	rm -f sssp
