ICPC = icpc

ifeq (${debug}, 1)
	CC_FLAGS = -Wall -Wextra -g -std=c++11 -qopenmp -DONEDEBUG
else
	CC_FLAGS = -Wall -Wextra -g -std=c++11 -qopenmp -O3 -qopt-report=5 -qopt-report-phase=vec
endif


ifeq (${symmetric}, 1)
	CC_FLAGS += -DONESYMMETRIC
endif

.PHONY: clean

all: page_rank

page_rank: page_rank.cpp
#icc -mmic -O3 -o page_rank page_rank.cpp -w
	$(ICPC) $(CC_FLAGS) $^ -o $@

clean:
	rm -f page_rank
