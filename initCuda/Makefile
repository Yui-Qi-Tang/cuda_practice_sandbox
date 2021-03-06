# versioning
MAJOR=0
MINOR=1
PATCH=0
VERSION=$(MAJOR).$(MINOR).$(PATCH)

BIN=$(notdir $(CURDIR))
DBG=$(BIN)g

SRC=src
INC=inc
OBJ=obj/bin
DBJ=obj/dbg
BLD=bin
LIB=lib

INCLUDES=-I. -I$(INC)

CXX=g++
CXXFLAGS= -O2 -std=c++11 -lpthread -pthread

# g++
SOURCES=$(wildcard $(SRC)/*.cpp)
OBJS=$(notdir $(patsubst %cpp,%o,$(SOURCES)))
BIN_OBJS=$(addprefix $(OBJ)/,$(OBJS))

# nvcc
NVCC=nvcc
NVCCFLAGS= -O2 -lm -std=c++11 -Wno-deprecated-gpu-targets --ptxas-options=-v -D_FORCE_INLINES
NVCCDEBUG= -g -O0

# nvcc build rule
CUSOURCES=$(wildcard $(SRC)/*.cu)
CUOBJS=$(notdir $(patsubst %cu,%o,$(CUSOURCES)))
CUBIN_OBJS=$(addprefix $(OBJ)/,$(CUOBJS))

# rule to build g++ .o
$(OBJ)/%.o: $(SRC)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
	# $(NVCC) $(NVCCFLAG) $(INCLUDES) -c $< -o $@

# rule to build nvcc .o
$(OBJ)/%.o: $(SRC)/%.cu
	$(NVCC) $(NVCCFLAG) $(INCLUDES) -c $< -o $@

.PHONY: clean

# pre_check -> $(BIN_OBJS): hit rule to build c++ .o -> $(BIN_OBJS): hit rule build nvcc .o -> $(BLD)/$(BIN):hit rule to build binary
all: pre_check $(BIN_OBJS) $(CUBIN_OBJS) $(BLD)/$(BIN)

# rule to build binary
$(BLD)/$(BIN): $(BIN_OBJS) $(CUBIN_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $(BLD)/$(BIN) $(BIN_OBJS) $(CUBIN_OBJS)

pre_check:
	@if ! test -f $(OBJ); then mkdir -p $(OBJ); fi
	@if ! test -f $(DBJ); then mkdir -p $(DBJ); fi
	@if ! test -f $(BLD); then mkdir -p $(BLD); fi

clean:
	rm -f $(CUBIN_OBJS) $(BIN_OBJS) $(BLD)/* core
