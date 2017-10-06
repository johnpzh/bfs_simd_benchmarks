ICPC = icpc

ifeq (${debug}, 1)
	CC_FLAGS = -Wall -Wextra -g -std=c++11 -qopenmp -DONEDEBUG
else
	CC_FLAGS = -Wall -Wextra -g -std=c++11 -qopenmp -O3 -xMIC-AVX512 -qopt-report=5 -qopt-report-phase=vec -inline-level=2 -inline-forceinline
endif

ifeq (${serial}, 1)
	CC_FLAGS += -DONESERIAL
endif

CC_FLAGS += -lmemkind

.PHONY: clean

all: cc

cc: cc.cpp
	$(ICPC) $(CC_FLAGS) -o $@ $^

clean:
	rm -f cc
