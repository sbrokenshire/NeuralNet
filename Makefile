CFLAGS+=-Wall -Wextra -std=c99 -pedantic

nn.o: nn.h
nn.tests.o: nn.h

backprop.o: backprop.h nn.h
backprop.tests.o: backprop.h nn.h

nn.tests: nn.o nn.tests.o
	$(CC) $(CFLAGS) $(LDFLAGS) nn.o nn.tests.o -o nn.tests

backprop.tests: backprop.o backprop.tests.o nn.o
	$(CC) $(CFLAGS) $(LDFLAGS) backprop.o backprop.tests.o nn.o -o backprop.tests

clean:
	rm -f *.o *.tests

.PHONY: clean 
