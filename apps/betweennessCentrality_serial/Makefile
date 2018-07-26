ICPC = icpc

ifeq (${debug}, 1)
	CPPFLAGS = -Wall -Wextra -g -std=c++14 -qopenmp -DONEDEBUG
else
	CPPFLAGS = -Wall -Wextra -g -std=c++14 -qopenmp -O3 -qopt-report=5 -qopt-report-phase=vec
endif

INCLUDE = -I/home/zpeng/lib/include
LIB = /home/zpeng/lib/lib/libpapi.a

.PHONY: clean

all: bc

bc: bc.cpp
	$(ICPC) $(CPPFLAGS) $(INCLUDE) -o $@ $^ $(LIB)

clean:
	rm -f bc
