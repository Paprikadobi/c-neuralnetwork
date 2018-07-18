main:
	@gcc $(shell find . ! -name "Test.c" -name "*.c") -o Main.o -O3 -std=c99
test:
	@gcc Matrix.c Test.c -o Test.o
debug:
	@gcc $(shell find . ! -name "Test.c" -name "*.c") -o Main-debug.o -g
clear:
	@rm -rf *.o*
