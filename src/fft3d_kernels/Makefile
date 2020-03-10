# Author : Arjun Ramaswami
# email  : ramaswami.arjun@gmail.com
#
# Multi Step Compilation for Intel OpenCL Kernel Code

# Check if Intel OpenCL SDK is installed and path set
ifeq ($(wildcard $(INTELFPGAOCLSDKROOT)),)
$(error Set INTELFPGAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation)
endif
ifeq ($(wildcard $(INTELFPGAOCLSDKROOT)/host/include/CL/opencl.h),)
$(error Set INTELFPGAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation.)
endif

# Variables
LOGN := 6
BOARD := p520_hpc_sg280l     # Stratix 10
#BOARD := p385a_sch_ax115    # Arria 10
FPGA_DP := 0

# Compilers
AOC := aoc

# Compiler Flags
AOCFLAGS   := -g -v -D LOGN=$(LOGN) -global-ring -duplicate-ring -cl-single-precision-constant -no-interleaving=default
FPFLAGS    :=
EMUFLAGS   := -march=emulator -legacy-emulator
SIMFLAGS   := -march=simulator 
INTERFLAGS := -report -rtl
PROFFLAGS  := -fp-relaxed -profile=all
SYNFLAGS   := -fp-relaxed

# debug flags are purposely appended to emuflags instead of aocflags 
# therefore, DEBUG mode only for emulation
# mostly used to check if the preprocessor flags enable correct code compilation
ifeq ($(DEBUG),1)
	EMUFLAGS := $(EMUFLAGS) -Werror -DDEBUG
else
	EMUFLAGS := $(EMUFLAGS)
endif	

# Libraries 

# Paths to targets
TARGET_PATH    := fpgabitstream/fft3d
TARGET_DP_DIR  := $(TARGET_PATH)/synthesis_dp
TARGET_SP_DIR  := $(TARGET_PATH)/synthesis_sp

# Pass FPGA_DP=1 as cmd line arg to make to activate double precision compilation 
# Only for emulation and for generating reports
# double precision disables relaxation and rounding in floating point operations 
ifeq ($(FPGA_DP),1)
	TARGET_DIR := $(TARGET_DP_DIR)
 	FPFLAGS :=  $(FPFLAGS) -fpc
else
	TARGET_DIR := $(TARGET_SP_DIR)
	FPFLAGS := $(FPFLAGS) -fp-relaxed -D __FPGA_SP
endif

TARGET_EMU_DIR   ?= $(TARGET_DIR)/emulation/emu$(LOGN)
TARGET_SIM_DIR   ?= $(TARGET_DIR)/simulation/sim$(LOGN)
TARGET_INTER_DIR ?= $(TARGET_DIR)/report/rep$(LOGN)
TARGET_PROF_DIR  ?= $(TARGET_DIR)/profile/prof$(LOGN)
TARGET_SYN_DIR   ?= $(TARGET_DIR)/synthesis/syn$(LOGN)
	
# Files
KERNEL_SRCS := ./fft3d.cl

# Targets
TARGET_KERNEL := fft3d.aocx
TARGET_INTERMEDIATE := fft3d.aocr

default: help 
	$(error No target specified)

help: 
	$(info ***************************************************)
	$(info Choose one of the steps to compile the FFT3d kernel)
	$(info ***************************************************)
	$(info emulate  =  Compile kernels for emulation)
	$(info syn      =  Synthesize kernels  )
	$(info inter    =  Intermediate Report )
	$(info profile  =  Profile kernels  )
	$(info sim      =  Simulate kernels )
	$(info clean    =  deletes syn folders and files)
	$(info emuclean =  deletes emulation folders and files)
	$(info fullclean = deletes all files )
	$(info help     =  Prints help )
	$(info ***************************************************)
	$(info Command Line Flags with Make:					  )
	$(info ***************************************************)
	$(info LOGN=(4:5:6) = select the size of the fft to synthesize, default is 6. )
	$(info FPGA_DP=1   = enables emulation with double precision floating operations)
	$(info DEBUG=1     = enables debug messages for emulation)
	$(info ***************************************************)

# Kernel Emulation
emulate: info
	mkdir -p $(TARGET_EMU_DIR)
	aoc $(FPFLAGS) $(AOCFLAGS) $(EMUFLAGS) -board=$(BOARD) $(KERNEL_SRCS) -o $(TARGET_EMU_DIR)/$(TARGET_KERNEL)

# kernel intermediate compilation for report
inter: info
	mkdir -p $(TARGET_INTER_DIR)
	aoc $(FPFLAGS) $(AOCFLAGS) $(INTERFLAGS) -board=$(BOARD) $(KERNEL_SRCS) -o $(TARGET_INTER_DIR)/$(TARGET_INTERMEDIATE)

# Kernel Simulation - Only single precision floating point ops
sim: info 
	mkdir -p $(TARGET_SIM_DIR)
	aoc -D __FPGA_SP $(AOCFLAGS) $(SIMFLAGS) $(KERNEL_SRCS) -o $(TARGET_SIM_DIR)/$(TARGET_KERNEL)

# Kernel Profiling - Only single precision floating point ops
profile: info 
	mkdir -p $(TARGET_PROF_DIR)
	aoc -D __FPGA_SP $(AOCFLAGS) $(PROFFLAGS) --board $(BOARD) $(KERNEL_SRCS) -o $(TARGET_PROF_DIR)/$(TARGET_KERNEL) 
	#@echo aoc -D __FPGA_SP $(AOCFLAGS) $(PROFFLAGS) --board $(BOARD) $(KERNEL_SRCS) -o $(TARGET_PROF_DIR)/$(TARGET_KERNEL) > profile_$$(date +%Y%m%d%H%M%S).log 2>&1

# Kernel Synthesis - Only single precision floating point ops
syn: info 
	mkdir -p $(TARGET_SYN_DIR)
	aoc -D __FPGA_SP $(AOCFLAGS) $(SYNFLAGS) --board $(BOARD) $(KERNEL_SRCS) -o $(TARGET_SYN_DIR)/$(TARGET_KERNEL) 
	#@echo aoc -D __FPGA_SP $(AOCFLAGS) $(SYNFLAGS) --board $(BOARD) $(KERNEL_SRCS) -o $(TARGET_SYN_DIR)/$(TARGET_KERNEL) > syn_$$(date +%Y%m%d%H%M%S).log 2>&1 
	#nohup aoc -fp-relaxed -g -v $(KERNEL_SRCS) -o $@ > test.log &

clean:
	rm -rf d$(TARGET_SYN_DIR)/
emuclean: 
	rm -rf $(TARGET_EMU_DIR)/
fullclean: 
	rm -rf $(TARGET_PATH)/

info: 
	$(info -------------------------------)
	$(info QUARTUS VERSION    : $(QUARTUS_VERSION))
	$(info BOARD              : $(BOARD))
	$(info LOG FFT SIZE       : $(LOGN))
	$(info PRECISION          : $(FPGA_DP) (0/sp 1/dp))
	$(info -------------------------------)

.PHONY : clean emuclean fullclean syn profile sim inter emulate help default info # ---------------
