print-%: ; @echo $*=$($*)

PTPACK_HOME = ..

include_dirs = $(PTPACK_HOME)/include $(PTPACK_HOME)/3rd-party

lib_dir = $(PTPACK_HOME)/build/lib
ptpack_lib = $(lib_dir)/libptpack.a

CXXFLAGS := -Wall -O3 -m64 -std=c++11
CXXFLAGS += $(addprefix -I,$(include_dirs))

CXX := g++

SRCS = $(wildcard *.cc)
PROGS = $(patsubst %.cc,%,$(SRCS))

build_dir = build

TARGET = $(patsubst %,$(build_dir)/%,$(PROGS))

all: $(TARGET)

$(build_dir)/%: %.cc $(ptpack_lib)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(lib_dir) -lptpack

clean:
	rm -rf $(TARGET)

-include $(DEPS)
