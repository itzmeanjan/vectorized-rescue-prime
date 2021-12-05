CXX = gcc
CXX_FLAGS = -std=c11 -Wall
INCLUDE_DIR = -I./include
LINK_FLAGS = -lOpenCL
PROG = run

$(PROG): main.o utils.o
	$(CXX) $^ $(LINK_FLAGS) -o $@

utils.o: utils.c
	$(CXX) $(CXX_FLAGS) $(INCLUDE_DIR) -c $^

main.o: main.c
	$(CXX) $(CXX_FLAGS) $(INCLUDE_DIR) -c $^

clean:
	find . -name "*.o" -o -name "a.out" -o -name "run" | xargs rm -f
