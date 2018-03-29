CC = icpc
CPPFLAGS = -Wall

.PHONY: all clean

all: tile-size-computer

tile-size-computer: tile-size-computer.cpp
	$(CC) $(CPPFLAGS) -o $@ $^

clean:
	rm -f tile-size-computer
