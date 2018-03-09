ICPC = icpc

ifeq (${debug}, 1)
	CPPFLAGS = -Wall -Wextra -g -std=c++14 -qopenmp -DONEDEBUG
else
	CPPFLAGS = -Wall -Wextra -g -std=c++14 -qopenmp -O3 -qopt-report=5 -qopt-report-phase=vec
endif

ifeq (${weighted}, 1)
	CPPFLAGS += -DWEIGHTED
endif

.PHONY: clean

all: page_rank

page_rank: page_rank.cpp
#icc -mmic -O3 -o page_rank page_rank.cpp -w
	$(ICPC) $(CPPFLAGS) $^ -o $@

clean:
	rm -f page_rank
