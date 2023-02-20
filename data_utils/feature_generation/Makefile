CC = g++
PY3_INC = $(shell python3 -c 'import sys; print("/usr/include/python" + str(sys.version_info.major) + "." + str(sys.version_info.minor))')
NP_INC = $(shell python3 -c "import numpy; print(numpy.__path__[0])")/core/include/

all: path_algorithms.so

path_algorithms.so:
	cython -3 path_algorithms.pyx -o path_algorithms.c
	$(CC) path_algorithms.c \
		-shared \
		-fPIC \
		-I$(PY3_INC) \
		-I$(NP_INC) \
		-D NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION \
		-o path_algorithms.so
	rm path_algorithms.c

clean:
	rm path_algorithms.so
