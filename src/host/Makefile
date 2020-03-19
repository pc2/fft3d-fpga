# Author : Arjun Ramaswami
# email  : ramaswami.arjun@gmail.com
#
# Compiles Host Code used to run FFT3d kernels 

# Check if Intel OpenCL SDK is installed and path set
ifeq ($(wildcard $(INTELFPGAOCLSDKROOT)),)
$(error Set INTELFPGAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation)
endif
ifeq ($(wildcard $(INTELFPGAOCLSDKROOT)/host/include/CL/opencl.h),)
$(error Set INTELFPGAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation.)
endif

# Variables
DEBUG := 0

# Compilers
CC := gcc -std=c11

# Compiler Flags
CFLAGS := -march=native -Wall

# Flags
ifeq ($(DEBUG),1)
	CFLAGS := $(CFLAGS) -O1 -fsanitize=leak -DDEBUG -ggdb3
else
	CFLAGS := $(CFLAGS) -O3 -funroll-loops -mtune=native
endif

ifeq ($(VERBOSE),1)
	CFLAGS := $(CFLAGS) -DVERBOSE
endif

SRCS := main.c $(wildcard src/*.c ext/src/*.c)

AOCL_HEADER_DIRS := $(shell aocl compile-config)
INC := -Iinclude $(AOCL_HEADER_DIRS) -Iext/include

AOCL_LINKER_DIRS = $(shell aocl link-config) 
LIBS := $(AOCL_LINKER_DIRS)

# Target
TARGETDIR := ../bin
TARGET := host

all : 
	mkdir -p $(TARGETDIR)
	$(CC) $(SRCS) $(CFLAGS) $(INC) $(LIBS) -o $(TARGETDIR)/$(TARGET)

clean : 
	rm -f $(TARGETDIR)/$(TARGET)

.PHONY : all clean # ---------------
