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

all: kcore

kcore: kcore.cpp
	$(ICPC) $(CPPFLAGS) -o $@ $^

clean:
	rm -f kcore
