# C compiler
ICPC = icpc
CPPFLAGS = -Wall -Wextra -g -std=c++14


TARGET =  prune

.PHONY: clean, all

all: $(TARGET)

prune: prune.cpp
	$(ICPC) $(CPPFLAGS) $^ -o $@

clean:
	rm -f $(TARGET)
