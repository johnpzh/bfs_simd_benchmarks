# C compiler
ICPC = icpc

ifeq (${debug}, 1)
	CC_FLAGS = -Wall -Wextra -g -std=c++14 -qopenmp -DONEDEBUG -O0
else
	CC_FLAGS = -Wall -Wextra -g -std=c++14 -qopenmp -O3 -xMIC-AVX512 -qopt-report=5 -qopt-report-phase=vec -inline-level=2 -inline-forceinline
endif

#INCLUDE = -I/home/zpeng/lib/include
#LIB = /home/zpeng/lib/lib/libpapi.a

.PHONY: clean, all

all: bc

bc: bc.cpp 
	$(ICPC) $(CC_FLAGS) $(INCLUDE) -o $@ $^ $(LIB)

clean:
	rm -f bc
