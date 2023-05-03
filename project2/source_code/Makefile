CC=gcc
CFLAGS=-Wall -O3 -pthread -lm


SRC=jacobi_solver.c compute_gold.c

# Object files
OBJ=$(SRC:.c=.o)

# Name of the output binary
TARGET=jacobi_solver

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
