CXX=g++ 
CXX_FLAGS=-std=c++11 -O3 -march=native
YANNSA_INC=src/include/

GTEST_PATH=/usr/local
GTEST_LIB=$(GTEST_PATH)/lib/
GTEST_INC=$(GTEST_PATH)/include/

UNIT_TEST=yannsa_test
UNIT_TEST_LDFLAGS=$(LDFLAGS) -L$(GTEST_LIB) -lgtest -lgtest_main
UNIT_TEST_ROOT=test
UNIT_TEST_OBJ_ROOT=build_$(UNIT_TEST_ROOT)
UNIT_TEST_SRC=$(wildcard $(UNIT_TEST_ROOT)/*.cc)
UNIT_TEST_OBJ=$(patsubst $(UNIT_TEST_ROOT)%.cc, $(UNIT_TEST_OBJ_ROOT)%.o, $(UNIT_TEST_SRC))

$(UNIT_TEST_OBJ_ROOT)/%.o: $(UNIT_TEST_ROOT)/%.cc
	@mkdir -p $(@D)
	$(CXX) $(UNIT_TEST_CFLAGS) -I$(GTEST_INC) -I$(YANNSA_INC) -o $@ -c $<

$(UNIT_TEST): $(UNIT_TEST_OBJ) 
	$(CXX) $(UNIT_TEST_CFLAGS) -o $@ $^ $(UNIT_TEST_LDFLAGS)

