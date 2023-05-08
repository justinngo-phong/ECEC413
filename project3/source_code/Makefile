C = gcc
CFLAGS = -O3 -Wall -lpthread -lm

SRCS = gauss_eliminate.c compute_gold.c
OBJS = $(SRCS:.c=.o)

.PHONY: all clean

all: gauss_eliminate

gauss_eliminate: $(OBJS)
		$(CC) $(CFLAGS) -o gauss_eliminate $(OBJS)

%.o: %.c
		$(CC) $(CFLAGS) -c $< -o $@

clean:
		rm -f gauss_eliminate $(OBJS)
