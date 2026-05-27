SHELL := /bin/bash

PROJECT_ROOT := $(CURDIR)
BUILD_DIR ?= $(PROJECT_ROOT)/build
CUDA_HOME ?= /usr/local/cuda
CC ?= gcc
CXX ?= g++
NVCC ?= $(CUDA_HOME)/bin/nvcc
AR ?= ar
PKG_CONFIG ?= pkg-config

CUDA_ARCH ?= 86
CUDA_ARCH_FLAGS ?= -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) -gencode arch=compute_$(CUDA_ARCH),code=compute_$(CUDA_ARCH)
CUDA_DETECTED_MAJOR := $(shell $(NVCC) --version 2>/dev/null | sed -n 's/^.*release \([0-9][0-9]*\)\..*$$/\1/p' | head -1)
CUDA_SO_MAJOR ?= $(if $(CUDA_DETECTED_MAJOR),$(CUDA_DETECTED_MAJOR),13)
CUDA_VERSION ?= $(CUDA_SO_MAJOR).0.0
CUDA_ALLOW_UNSUPPORTED_COMPILER ?= 1
STATIC_CXX_RUNTIME ?= 0
JOBS ?= $(shell nproc 2>/dev/null || echo 1)

ifneq ($(shell command -v g++-13 2>/dev/null),)
CUDA_HOST_COMPILER ?= g++-13
else
CUDA_HOST_COMPILER ?= $(CXX)
endif

COMMON_INC := $(PROJECT_ROOT)/common/include
CUDA_INC := $(CUDA_HOME)/include
CUDA_LIB := $(CUDA_HOME)/lib64
BOOST_CPPFLAGS ?=
BOOST_LIBS ?= -lboost_program_options
CUDA_DRIVER_LIBS ?= -lnvidia-ml -lcuda

CPPFLAGS += -I$(COMMON_INC)
CFLAGS += -O2 -g -fPIC -Wall -Wextra
CXXFLAGS += -O2 -g -fPIC -Wall -Wextra -std=c++17
NVCCFLAGS += -std=c++17 $(CUDA_ARCH_FLAGS) -ccbin $(CUDA_HOST_COMPILER) -I$(COMMON_INC) -Xcompiler -fPIC
SAMPLE_NVCCFLAGS += -std=c++17 $(CUDA_ARCH_FLAGS) -ccbin $(CUDA_HOST_COMPILER) -I$(COMMON_INC) -cudart shared
SAMPLE_LDFLAGS += -L$(CUDA_LIB)
ifeq ($(STATIC_CXX_RUNTIME),1)
SAMPLE_LDFLAGS += -Xcompiler -static-libstdc++ -Xcompiler -static-libgcc
endif
ifeq ($(CUDA_ALLOW_UNSUPPORTED_COMPILER),1)
NVCCFLAGS += -allow-unsupported-compiler
SAMPLE_NVCCFLAGS += -allow-unsupported-compiler
endif

GUEST_TRANSPORT_DIR := $(BUILD_DIR)/guest/transport
GUEST_INTERCEPT_DIR := $(BUILD_DIR)/guest/intercept
HOST_VHOST_DIR := $(BUILD_DIR)/host/vhost
SAMPLE_DIR := $(BUILD_DIR)/sample
THIRD_PARTY_DIR := $(BUILD_DIR)/third_party

TRANSPORT_OBJ := $(GUEST_TRANSPORT_DIR)/hmcuda_transport.o
TRANSPORT_LIB := $(GUEST_TRANSPORT_DIR)/libhmcuda_transport.a

RUNTIME_LIB := $(GUEST_INTERCEPT_DIR)/libcudart.so.$(CUDA_VERSION)
RUNTIME_VERSION_SCRIPT := $(GUEST_INTERCEPT_DIR)/libcudart.version
DRIVER_LIB := $(GUEST_INTERCEPT_DIR)/libcuda.so.1.0.0
NVML_LIB := $(GUEST_INTERCEPT_DIR)/libnvidia-ml.so.1.0.0
VHOST_BIN := $(HOST_VHOST_DIR)/vhost_user_hmcuda
SAMPLES := $(SAMPLE_DIR)/matrixMul $(SAMPLE_DIR)/smallTest

LIBVHOST_USER_SRC_DIR := third_party/libvhost-user
VHOST_SRCS := \
	host/vhost/vhost_user_hmcuda.c \
	host/vhost/cmd_runtime.c \
	host/vhost/cmd_driver.c \
	host/vhost/core_runtime.c \
	host/vhost/core_driver.c \
	host/vhost/resource.c

ifneq ($(wildcard $(LIBVHOST_USER_SRC_DIR)/libvhost-user.c),)
VHOST_DEPS_AVAILABLE := 1
else
VHOST_DEPS_AVAILABLE := 0
endif

VHOST_OBJS := $(patsubst host/vhost/%.c,$(HOST_VHOST_DIR)/%.o,$(VHOST_SRCS))
LIBVHOST_USER_DIR := $(BUILD_DIR)/third_party/libvhost-user
LIBVHOST_USER_SRCS := \
	$(LIBVHOST_USER_SRC_DIR)/libvhost-user.c \
	$(LIBVHOST_USER_SRC_DIR)/libvhost-user-glib.c
LIBVHOST_USER_OBJS := $(patsubst $(LIBVHOST_USER_SRC_DIR)/%.c,$(LIBVHOST_USER_DIR)/%.o,$(LIBVHOST_USER_SRCS))
LIBVHOST_USER_LIB := $(LIBVHOST_USER_DIR)/libvhost-user.a

NVBANDWIDTH_SRC_DIR := third_party/nvbandwidth
NVBANDWIDTH_BIN := $(THIRD_PARTY_DIR)/nvbandwidth/nvbandwidth
NVBANDWIDTH_SRCS := \
	$(NVBANDWIDTH_SRC_DIR)/testcase.cpp \
	$(NVBANDWIDTH_SRC_DIR)/testcases_ce.cpp \
	$(NVBANDWIDTH_SRC_DIR)/testcases_sm.cpp \
	$(NVBANDWIDTH_SRC_DIR)/kernels.cu \
	$(NVBANDWIDTH_SRC_DIR)/memcpy.cpp \
	$(NVBANDWIDTH_SRC_DIR)/nvbandwidth.cpp \
	$(NVBANDWIDTH_SRC_DIR)/multinode_memcpy.cpp \
	$(NVBANDWIDTH_SRC_DIR)/multinode_testcases.cpp \
	$(NVBANDWIDTH_SRC_DIR)/output.cpp \
	$(NVBANDWIDTH_SRC_DIR)/json_output.cpp \
	$(NVBANDWIDTH_SRC_DIR)/json/jsoncpp.cpp
NVBANDWIDTH_VERSION := $(shell git -C $(NVBANDWIDTH_SRC_DIR) describe --always --tags 2>/dev/null || echo unknown)

GLIB_CFLAGS := $(shell $(PKG_CONFIG) --cflags glib-2.0 2>/dev/null)
GLIB_LIBS := $(shell $(PKG_CONFIG) --libs glib-2.0 2>/dev/null)

.DEFAULT_GOAL := all

.PHONY: all guest host samples benchmarks nvbandwidth driver clean clean-guest clean-host clean-samples help
.PHONY: hmcuda_runtime hmcuda_driver hmcuda_nvml vhost_user_hmcuda matrixMul smallTest

all: guest host samples benchmarks

guest: hmcuda_runtime hmcuda_driver hmcuda_nvml
host: vhost_user_hmcuda
samples: matrixMul smallTest
benchmarks: nvbandwidth

hmcuda_runtime: $(RUNTIME_LIB)
hmcuda_driver: $(DRIVER_LIB)
hmcuda_nvml: $(NVML_LIB)
vhost_user_hmcuda: $(VHOST_BIN)
matrixMul: $(SAMPLE_DIR)/matrixMul
smallTest: $(SAMPLE_DIR)/smallTest

$(GUEST_TRANSPORT_DIR) $(GUEST_INTERCEPT_DIR) $(HOST_VHOST_DIR) $(SAMPLE_DIR) $(THIRD_PARTY_DIR):
	@mkdir -p $@

$(TRANSPORT_OBJ): guest/transport/hmcuda_transport.cpp guest/transport/hmcuda_transport.h guest/transport/hmcuda_log.h common/include/hmcuda_api.h | $(GUEST_TRANSPORT_DIR)
	$(CXX) $(CPPFLAGS) -Iguest/transport -I$(CUDA_INC) $(CXXFLAGS) -c $< -o $@

$(TRANSPORT_LIB): $(TRANSPORT_OBJ)
	$(AR) rcs $@ $^

$(GUEST_INTERCEPT_DIR)/hmcuda_runtime.o: guest/intercept/hmcuda_runtime.cpp $(TRANSPORT_LIB) | $(GUEST_INTERCEPT_DIR)
	$(CXX) $(CPPFLAGS) -Iguest/transport -Iguest/intercept -I$(CUDA_INC) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_VERSION_SCRIPT): Makefile | $(GUEST_INTERCEPT_DIR)
	printf 'libcudart.so.%s {\n    global: *;\n};\n' '$(CUDA_SO_MAJOR)' > $@

$(RUNTIME_LIB): $(GUEST_INTERCEPT_DIR)/hmcuda_runtime.o $(TRANSPORT_LIB) $(RUNTIME_VERSION_SCRIPT)
	$(CXX) -shared -Wl,-soname,libcudart.so.$(CUDA_SO_MAJOR) -Wl,--version-script=$(RUNTIME_VERSION_SCRIPT) $< $(TRANSPORT_LIB) -o $@
	ln -sf libcudart.so.$(CUDA_VERSION) $(GUEST_INTERCEPT_DIR)/libcudart.so.$(CUDA_SO_MAJOR)
	ln -sf libcudart.so.$(CUDA_SO_MAJOR) $(GUEST_INTERCEPT_DIR)/libcudart.so

$(GUEST_INTERCEPT_DIR)/hmcuda_driver.o: guest/intercept/hmcuda_driver.cpp $(TRANSPORT_LIB) | $(GUEST_INTERCEPT_DIR)
	$(CXX) $(CPPFLAGS) -Iguest/transport -Iguest/intercept -I$(CUDA_INC) $(CXXFLAGS) -c $< -o $@

$(DRIVER_LIB): $(GUEST_INTERCEPT_DIR)/hmcuda_driver.o $(TRANSPORT_LIB)
	$(CXX) -shared -Wl,-soname,libcuda.so.1 $< $(TRANSPORT_LIB) -o $@
	ln -sf libcuda.so.1.0.0 $(GUEST_INTERCEPT_DIR)/libcuda.so.1
	ln -sf libcuda.so.1 $(GUEST_INTERCEPT_DIR)/libcuda.so

$(GUEST_INTERCEPT_DIR)/hmcuda_nvml_stub.o: guest/intercept/hmcuda_nvml_stub.c | $(GUEST_INTERCEPT_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(NVML_LIB): $(GUEST_INTERCEPT_DIR)/hmcuda_nvml_stub.o
	$(CC) -shared -Wl,-soname,libnvidia-ml.so.1 $< -o $@
	ln -sf libnvidia-ml.so.1.0.0 $(GUEST_INTERCEPT_DIR)/libnvidia-ml.so.1
	ln -sf libnvidia-ml.so.1 $(GUEST_INTERCEPT_DIR)/libnvidia-ml.so

$(HOST_VHOST_DIR)/%.o: host/vhost/%.c
	@mkdir -p $(dir $@)
	@if [ -z "$(GLIB_CFLAGS)" ]; then echo "glib-2.0 not found by pkg-config; install GLib development files or set PKG_CONFIG_PATH" >&2; exit 1; fi
	$(CC) $(CPPFLAGS) -I$(CUDA_INC) -I$(LIBVHOST_USER_SRC_DIR) $(GLIB_CFLAGS) $(CFLAGS) -D_GNU_SOURCE -c $< -o $@

$(LIBVHOST_USER_DIR)/%.o: $(LIBVHOST_USER_SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	@if [ -z "$(GLIB_CFLAGS)" ]; then echo "glib-2.0 not found by pkg-config; install GLib development files or set PKG_CONFIG_PATH" >&2; exit 1; fi
	$(CC) -I$(LIBVHOST_USER_SRC_DIR) $(GLIB_CFLAGS) $(CFLAGS) -D_GNU_SOURCE -c $< -o $@

$(LIBVHOST_USER_LIB): $(LIBVHOST_USER_OBJS)
	$(AR) rcs $@ $^

ifeq ($(VHOST_DEPS_AVAILABLE),1)
$(VHOST_BIN): $(VHOST_OBJS) $(LIBVHOST_USER_LIB)
	$(CC) $(VHOST_OBJS) $(LIBVHOST_USER_LIB) $(GLIB_LIBS) -L$(CUDA_LIB) -lcudart -lcuda -lpthread -o $@
else
$(VHOST_BIN):
	@echo "third_party/libvhost-user is not available; cannot build vhost_user_hmcuda" >&2
	@echo "Initialize submodules with: git submodule update --init third_party/libvhost-user" >&2
	@exit 1
endif

$(SAMPLE_DIR)/matrixMul: sample/matrixMul.cu | $(SAMPLE_DIR)
	$(NVCC) $(SAMPLE_NVCCFLAGS) $(SAMPLE_LDFLAGS) $< -o $@ $(if $(USE_CUBLAS),-DUSE_CUBLAS -lcublas,)

$(SAMPLE_DIR)/smallTest: sample/smallTest.cu | $(SAMPLE_DIR)
	$(NVCC) $(SAMPLE_NVCCFLAGS) $(SAMPLE_LDFLAGS) $< -o $@

nvbandwidth: $(NVBANDWIDTH_BIN)

$(NVBANDWIDTH_BIN): $(NVBANDWIDTH_SRCS) Makefile | $(THIRD_PARTY_DIR)
	@if [ ! -f "$(NVBANDWIDTH_SRC_DIR)/nvbandwidth.cpp" ]; then echo "third_party/nvbandwidth is not available; run git submodule update --init third_party/nvbandwidth" >&2; exit 1; fi
	@if [ -z "$(BOOST_CPPFLAGS)" ] && [ ! -f /usr/include/boost/program_options.hpp ]; then echo "Boost program_options headers not found; install Boost development files or set BOOST_CPPFLAGS/BOOST_LIBS" >&2; exit 1; fi
	@mkdir -p $(dir $@)
	$(NVCC) $(SAMPLE_NVCCFLAGS) $(BOOST_CPPFLAGS) -I$(NVBANDWIDTH_SRC_DIR) -I$(NVBANDWIDTH_SRC_DIR)/json -I$(CUDA_INC) -DGIT_VERSION=\"$(NVBANDWIDTH_VERSION)\" $(NVBANDWIDTH_SRCS) -o $@ -L$(CUDA_LIB) $(BOOST_LIBS) $(CUDA_DRIVER_LIBS)

driver:
	@echo "virtio_hmcuda is built in-tree by the external Linux kernel project." >&2
	@echo "This repository does not build the kernel driver." >&2
	@exit 1

clean: clean-guest clean-host clean-samples
	rm -rf $(THIRD_PARTY_DIR)

clean-guest:
	rm -rf $(BUILD_DIR)/guest

clean-host:
	rm -rf $(BUILD_DIR)/host

clean-samples:
	rm -rf $(BUILD_DIR)/sample

help:
	@echo "hmCUDA Makefile targets:"
	@echo "  all                 Build guest libraries, host daemon, samples, and benchmarks"
	@echo "  guest               Build intercept libraries"
	@echo "  host                Build vhost_user_hmcuda with libvhost-user"
	@echo "  samples             Build matrixMul and smallTest"
	@echo "  benchmarks          Build optional nvbandwidth"
	@echo "  clean               Remove Makefile build outputs"
	@echo ""
	@echo "Driver note:"
	@echo "  virtio_hmcuda is built in-tree by the external Linux kernel project."
	@echo ""
	@echo "CUDA settings:"
	@echo "  CUDA_HOME=/usr/local/cuda CUDA_ARCH=86 CUDA_ALLOW_UNSUPPORTED_COMPILER=1 STATIC_CXX_RUNTIME=0 USE_CUBLAS=1 make -j$(JOBS)"
