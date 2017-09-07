ICPC = icpc

ifeq (${debug}, 1)
	CC_FLAGS = -Wall -Wextra -g -std=c++11 -qopenmp -DONEDEBUG
else
	CC_FLAGS = -Wall -Wextra -g -std=c++11 -qopenmp -O3 -xMIC-AVX512 -qopt-report=5 -qopt-report-phase=vec
endif

.PHONY: clean

all: page_rank

page_rank: page_rank.cpp
	$(ICPC) $(CC_FLAGS) $^ -o $@

clean:
	rm -f page_rank
