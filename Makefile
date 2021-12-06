CXX = gcc
CXX_FLAGS = -std=c17 -Wall
INCLUDE_DIR = -I./include
LINK_FLAGS = -lOpenCL
PROG = run

$(PROG): main.o utils.o rescue_prime.o
	$(CXX) $^ $(LINK_FLAGS) -o $@

rescue_prime.o: rescue_prime.c include/rescue_prime.h include/rescue_prime_constants.h
	$(CXX) $(CXX_FLAGS) $(INCLUDE_DIR) -c $<

utils.o: utils.c include/utils.h
	$(CXX) $(CXX_FLAGS) $(INCLUDE_DIR) -c $<

main.o: main.c include/rescue_prime.h
	$(CXX) $(CXX_FLAGS) $(INCLUDE_DIR) -c $<

clean:
	find . -name "*.o" -o -name "*.gch" -o -name "a.out" -o -name "run" | xargs rm -f

format:
	find . -name "*.c" -o -name "*.h" -o -name "*.cl" | xargs clang-format -i
