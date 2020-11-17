# Makefile of the package
#
# Jichao Sun (js87@njit.edu)
# 
# Oct 3, 2013

CC	=	g++
CFLAGS	=	-Wall -I/afs/cad/linux/boost-1.60-sl6/include -fopenmp -ansi -ggdb -O3 -DTEST_CODE

P_SPBLAS=	nist_spblas
P_NNDES =       nndes

OBJ	=	$(P_SPBLAS)/nist_spblas.o \
		global.o \
		DenMatSin.o

BIN	=	franc

.PHONY: all bin clean
all: bin
bin: $(BIN)

################ $(BIN) #####################
franc:			franc.h \
			franc.cc \
			global.o \
			DenMatSin.o \
			$(P_NNDES)/*.h \
			$(P_SPBLAS)/nist_spblas.o
	$(CC) $(CFLAGS) -msse2 $(LIBS) -lm -o franc \
			franc.cc \
			global.o \
			DenMatSin.o \
			$(P_SPBLAS)/nist_spblas.o


################ $(OBJS) ####################
$(P_SPBLAS)/nist_spblas.o: \
			$(P_SPBLAS)/nist_spblas.cc \
			$(P_SPBLAS)/nist_spblas.h
	$(CC) $(CFLAGS) -o $(P_SPBLAS)/nist_spblas.o \
	    		-c $(P_SPBLAS)/nist_spblas.cc

global.o: 		global.cc \
			global.h
	$(CC) $(CFLAGS) -c global.cc

DenMatSin.o:		DenMatSin.cc \
			DenMatSin.h \
			global.h \
			$(P_SPBLAS)/nist_spblas.h
	$(CC) $(CFLAGS) -c DenMatSin.cc

################# clean ####################
clean:
	-rm -f $(BIN)
	-rm -f $(OBJ)
