main:
	@gcc $(shell find . ! -name "Test.c" -name "*.c") -o Main.o -O3 -std=c99
	@./Main.o
test:
	@gcc Matrix.c Test.c -o Test.o
	@./Test.o
clear:
	@rm *.o
