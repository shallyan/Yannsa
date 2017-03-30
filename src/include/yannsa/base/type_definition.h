#ifndef YANNSA_TYPE_DEFINITION_H
#define YANNSA_TYPE_DEFINITION_H 

#include <vector>
#include <unordered_set>

namespace yannsa {

typedef int IntIndex;

typedef int IntCode;

typedef std::vector<char> DynamicBitset;
typedef std::vector<IntIndex> IdList;
typedef std::unordered_set<IntIndex> IdSet;

} // namespace yannsa

#endif
