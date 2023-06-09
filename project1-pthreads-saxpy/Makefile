CC = gcc
CFLAGS = -O3 -Wall -std=c99
LDFLAGS = -lpthread -lm
SRC = saxpy.c
TARGET = saxpy

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)
