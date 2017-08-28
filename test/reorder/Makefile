ICPC = icpc

ifeq (${debug}, 1)
	CC_FLAGS = -Wall -Wextra -g -std=c++11 -qopenmp -DONEDEBUG
else
	CC_FLAGS = -Wall -Wextra -g -std=c++11 -qopenmp -O3 -qopt-report=5 -qopt-report-phase=vec
endif

ifeq (${untile}, 1)
	CC_FLAGS += -DUNTILE
endif

CC_FLAGS += -I/home/zpeng/lib/include/igraph -L/home/zpeng/lib/lib -ligraph

.PHONY: clean

all: page_rank

page_rank: page_rank.cpp
#icc -mmic -O3 -o page_rank page_rank.cpp -w
	$(ICPC) $(CC_FLAGS) $^ -o $@

clean:
	rm -f page_rank
