all: quake

CC = gcc
CFLAGS = -O2 -fopenmp

quake: quake.c
	$(CC) $(CFLAGS) -o quake $< -lm

clean:
	rm -f quake
