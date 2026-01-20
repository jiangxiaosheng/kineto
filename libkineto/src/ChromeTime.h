#include <chrono>
#include <cstdint>
#include "time_since_epoch.h"

namespace libkineto {
// std::chrono header start
#ifdef _GLIBCXX_USE_C99_STDINT_TR1
#define _KINETO_GLIBCXX_CHRONO_INT64_T int64_t
#elif defined __INT64_TYPE__
#define _KINETO_GLIBCXX_CHRONO_INT64_T __INT64_TYPE__
#else
#define _KINETO_GLIBCXX_CHRONO_INT64_T long long
#endif
// std::chrono header end

// There are tools like Chrome Trace Viewer that uses double to represent
// each element in the timeline. Double has a 53 bit mantissa to support
// up to 2^53 significant digits (up to 9007199254740992). This holds at the
// nanosecond level, about 3 months and 12 days. So, let's round base time to
// 3 months intervals, so we can still collect traces across ranks relative
// to each other.
// A month is 2629746, so 3 months is 7889238.
using _trimonths =
    std::chrono::duration<_KINETO_GLIBCXX_CHRONO_INT64_T, std::ratio<7889238>>;
#undef _GLIBCXX_CHRONO_INT64_T

class ChromeTraceBaseTime {
 public:
  ChromeTraceBaseTime() = default;
  static ChromeTraceBaseTime& singleton();
  void init() {
    get();
  }
  int64_t get() {
    // Make all timestamps relative to 3 month intervals.
    static int64_t base_time = libkineto::timeSinceEpoch(
        std::chrono::time_point<std::chrono::system_clock>(
            std::chrono::floor<_trimonths>(std::chrono::system_clock::now())));
    return base_time;
  }
};

// The 'ts' field written into the json file has 19 significant digits,
// while a double can only represent 15-16 digits. By using relative time,
// other applications can accurately read the 'ts' field as a double.
// Use the program loading time as the baseline time.
inline int64_t transToRelativeTime(int64_t time) {
  // Sometimes after converting to relative time, it can be a few nanoseconds
  // negative. Since Chrome trace and json processing will throw a parser error,
  // guard this.
  int64_t res = time - ChromeTraceBaseTime::singleton().get();
  if (res < 0) {
    return 0;
  }
  return res;
}
} // namespace libkineto
