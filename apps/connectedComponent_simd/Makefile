ICPC = icpc

ifeq (${debug}, 1)
	CPPFLAGS = -Wall -Wextra -g -std=c++14 -qopenmp -DONEDEBUG
else
	CPPFLAGS = -Wall -Wextra -g -std=c++14 -qopenmp -O3 -xMIC-AVX512 -qopt-report=5 -qopt-report-phase=vec -inline-level=2 -inline-forceinline
endif

#ifeq (${serial}, 1)
#	CPPFLAGS += -DONESERIAL
#endif

#INCLUDE = -I/home/zpeng/lib/include
#LIB = /home/zpeng/lib/lib/libpapi.a

.PHONY: clean

all: cc

cc: cc.cpp
	$(ICPC) $(CPPFLAGS) $(INCLUDE) -o $@ $^ $(LIB)

clean:
	rm -f cc
