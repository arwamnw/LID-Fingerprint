# Derived from the original Makefile of nndes for the standalone program
#
# Jichao Sun (js87@njit.edu)
#
# July 17, 2013
#   compile using -DTEST_CODE to use the bug-fixed version
#   otherwise, the original version
#
# July 16, 2013
#   I didn't see any point to use -lpthread
# July 12, 2013
#   -fopenmp and -lpthread can be removed to make it single threaded.
# July 11, 2013
#   Some instruction added to make nndes use DenMatSin class

CC=g++

# # original parallel options
# CXXFLAGS += -fopenmp -g -O3 -Wall -static -I. -msse2
# LDLIBS += -lboost_program_options  -lpthread -lm

# parallel options without -lpthread, plus -DTEST_CODE
# CXXFLAGS += -fopenmp -g -O3 -Wall -static -I. -msse2 -DTEST_CODE
CXXFLAGS += -fopenmp -g -O3 -Wall -static -I. -msse2
LDLIBS += -lboost_program_options  -lm

# # serial options
# CXXFLAGS += -g -O3 -Wall -static -I. -msse2
# LDLIBS += -lboost_program_options  -lm

.PHONY:	all clean

PROGS=nndes

all:	$(PROGS)

$(PROGS): *.cc *.h ../DenMatSin.cc ../DenMatSin.h
	$(CC) $(CXXFLAGS) nndes.cc ../DenMatSin.cc $(LDLIBS) -o nndes

clean:
	rm -f $(PROGS)

