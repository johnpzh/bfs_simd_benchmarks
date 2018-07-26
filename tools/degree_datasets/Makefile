ICPC = icpc
#CC_FLAGS = -Wall -Wextra -g -std=c++11 -qopenmp -O3 -qopt-report=5 -qopt-report-phase=vec
CC_FLAGS = -Wall -Wextra -g -std=c++11 -qopenmp -O3 -qopt-prefetch=5 -qopt-report=5 -qopt-report-phase=vec
#CC_FLAGS = -Wall -Wextra -g -std=c++11 -qopenmp

.PHONY: clean

all: page_rank

page_rank: page_rank.cpp
#icc -mmic -O3 -o page_rank page_rank.cpp -w
	$(ICPC) $(CC_FLAGS) $^ -o $@

clean:
	rm page_rank
