#pragma once

#include "ApproximateClock.h"
#include <array>

class ApproximateClockToUnixTimeConverter {
  public:
   ApproximateClockToUnixTimeConverter();
   std::function<time_t(libkineto::approx_time_t)> makeConverter();
 
   struct UnixAndApproximateTimePair {
     time_t t_;
     libkineto::approx_time_t approx_t_;
   };
   static UnixAndApproximateTimePair measurePair();
 
  private:
   static constexpr size_t replicates = 1001;
   using time_pairs = std::array<UnixAndApproximateTimePair, replicates>;
   time_pairs measurePairs();
 
   time_pairs start_times_;
 };

void register_time_converter();