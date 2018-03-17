ICPC = icpc

ifeq (${debug}, 1)
	CPPFLAGS = -Wall -Wextra -g -std=c++14 -qopenmp -DONEDEBUG
else
	CPPFLAGS = -Wall -Wextra -g -std=c++14 -qopenmp -O3 -qopt-report=5 -qopt-report-phase=vec
endif

ifeq (${serial}, 1)
	CPPFLAGS += -DONESERIAL
endif

.PHONY: clean

all: cc

cc: cc.cpp
	$(ICPC) $(CPPFLAGS) -o $@ $^

clean:
	rm -f cc
