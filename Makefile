CXX=g++-6
CXX_FLAGS=-std=c++11 -O3 -march=native -fopenmp

YANNSA_INC=src/include/
THIRD_PARTY_INC=third_party/

# example
EXAMPLE_FLAGS=$(CXX_FLAGS)
EXAMPLE_ROOT=example

index : $(EXAMPLE_ROOT)/index.cc
	$(CXX) $(EXAMPLE_FLAGS) -I $(YANNSA_INC) -I $(THIRD_PARTY_INC) -o $@ $^

extend_index : $(EXAMPLE_ROOT)/extend_index.cc
	$(CXX) $(EXAMPLE_FLAGS) -I $(YANNSA_INC) -I $(THIRD_PARTY_INC) -o $@ $^

dpg_index : $(EXAMPLE_ROOT)/dpg_index.cc
	$(CXX) $(EXAMPLE_FLAGS) -I $(YANNSA_INC) -I $(THIRD_PARTY_INC) -o $@ $^

search: $(EXAMPLE_ROOT)/search.cc
	$(CXX) $(EXAMPLE_FLAGS) -I $(YANNSA_INC) -I $(THIRD_PARTY_INC) -o $@ $^

bruteforce : $(EXAMPLE_ROOT)/bruteforce.cc
	$(CXX) $(EXAMPLE_FLAGS) -I $(YANNSA_INC) -I $(THIRD_PARTY_INC) -o $@ $^

# test
GTEST_PATH=/usr/local
GTEST_LIB=$(GTEST_PATH)/lib/
GTEST_INC=$(GTEST_PATH)/include/

UNIT_TEST_TARGET=yannsa_test
UNIT_TEST_FLAGS=$(CXX_FLAGS)
UNIT_TEST_LD_FLAGS=-L $(GTEST_LIB) -lgtest -lgtest_main
UNIT_TEST_ROOT=test
UNIT_TEST_OBJ_ROOT=build_$(UNIT_TEST_ROOT)
UNIT_TEST_SRC=$(wildcard $(UNIT_TEST_ROOT)/*.cc)
UNIT_TEST_OBJ=$(patsubst $(UNIT_TEST_ROOT)/%.cc, $(UNIT_TEST_OBJ_ROOT)/%.o, $(UNIT_TEST_SRC))

# $(warning $(UNIT_TEST_OBJ))

$(UNIT_TEST_OBJ_ROOT)/%.o: $(UNIT_TEST_ROOT)/%.cc
	@mkdir -p $(@D)
	$(CXX) $(UNIT_TEST_FLAGS) -I $(GTEST_INC) -I $(YANNSA_INC) -I $(THIRD_PARTY_INC) -o $@ -c $<

$(UNIT_TEST_TARGET): $(UNIT_TEST_OBJ) 
	$(CXX) $(UNIT_TEST_FLAGS) -o $@ $^ $(UNIT_TEST_LD_FLAGS)

