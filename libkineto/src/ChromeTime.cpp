#include "ChromeTime.h"

namespace libkineto {

ChromeTraceBaseTime& ChromeTraceBaseTime::singleton() {
  static ChromeTraceBaseTime instance;
  return instance;
}

} // namespace libkineto