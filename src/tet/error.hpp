#pragma once

#include <cstdio>
#include <cstdlib>

#include "tet/geometry.hpp"  // TET_HD

namespace tet {

// Library-level invariant violations: log and terminate (no exceptions). In device code the trap
// kills the kernel, and the next CUDA call on the host fails.
TET_HD inline void error( const char* message, const char* file, int line ) {
#ifdef __CUDA_ARCH__
  printf( "%s:%d: %s\n", file, line, message );
  __trap();
#else
  std::fprintf( stderr, "%s:%d: %s\n", file, line, message );
  std::abort();
#endif
}

}

#define ASSERT( cond, message ) do { if ( !( cond ) ) tet::error( message, __FILE__, __LINE__ ); } while ( 0 )
