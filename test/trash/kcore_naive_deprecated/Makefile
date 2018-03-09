ICPC = icpc

ifeq (${debug}, 1)
	CC_FLAGS = -Wall -Wextra -g -std=c++11 -qopenmp -DONEDEBUG
else
	CC_FLAGS = -Wall -Wextra -g -std=c++11 -qopenmp -O3 -qopt-report=5 -qopt-report-phase=vec
endif

ifeq (${serial}, 1)
	CC_FLAGS += -DONESERIAL
endif

.PHONY: clean

all: kcore

kcore: kcore.cpp
	$(ICPC) $(CC_FLAGS) -o $@ $^

clean:
	rm -f kcore
