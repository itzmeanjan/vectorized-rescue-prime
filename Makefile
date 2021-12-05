CXX = gcc
CXX_FLAGS = -std=c11 -Wall
LINK_FLAGS = -lOpenCL
PROG = run

$(PROG): main.o
	$(CXX) $^ $(LINK_FLAGS) -o $@

main.o: main.c
	$(CXX) $(CXX_FLAGS) -c $^

clean:
	find . -name "*.o" -o -name "a.out" -o -name "run" | xargs rm -f
