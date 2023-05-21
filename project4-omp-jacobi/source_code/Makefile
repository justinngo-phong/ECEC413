CC = gcc
CFLAGS = -fopenmp -std=c99 -Wall -O3 -lm
SRCS = jacobi_solver.c compute_gold.c
OBJS = $(SRCS:.c=.o)
EXECUTABLE = jacobi_solver

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS)

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	rm -f $(OBJS) $(EXECUTABLE)
