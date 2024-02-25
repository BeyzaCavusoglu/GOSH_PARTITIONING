CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include -I./
LIBS         := -lcusparse

all: row_per_block row_per_thread rows_per_thread row_per_warp

row_per_block: row_per_block.cu mmio.c smsh.c
	nvcc $(INC) row_per_block.cu mmio.c smsh.c -o row_per_block $(LIBS)

row_per_thread: row_per_thread.cu mmio.c smsh.c
	nvcc $(INC) row_per_thread.cu mmio.c smsh.c -o row_per_thread $(LIBS)
	
rows_per_thread: rows_per_thread.cu mmio.c smsh.c
	nvcc $(INC) rows_per_thread.cu mmio.c smsh.c -o rows_per_thread $(LIBS)

row_per_warp: row_per_warp.cu mmio.c smsh.c
	nvcc $(INC) row_per_warp.cu mmio.c smsh.c -o row_per_warp $(LIBS)

clean:
	rm -f row_per_block row_per_thread rows_per_thread row_per_warp

test:
	@echo "\n==== SpMV CSR Test ====\n"
	./row_per_block

.PHONY: clean all test
