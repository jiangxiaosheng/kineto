/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>
#include <string>

#include <libkineto.h>

// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "kineto_playground.cuh"
#include "time_converter.h"

using namespace kineto;

static const std::string kFileName =
    "/h/shengj2/code/kineto/libkineto/sample_programs/"
    "kineto_playground_trace.json";

int main() {
  std::cout << "Warm up " << std::endl;
  warmup();

  // Kineto config
  libkineto_init(false, true);

  // Kineto sets up a callback to convert gpu timestamp to tsc timestamp during collection
  // We need to register this converter to get back wall times during post processing
  // The code is completely stolen from pytorch
  register_time_converter();
  setOrcaMode(true);

  // Empty types set defaults to all types
  std::set<libkineto::ActivityType> types;

  // auto &profiler = libkineto::api().activityProfiler();
  // libkineto::api().initProfilerIfRegistered();
  // profiler.prepareTrace(types);

  // Good to warm up after prepareTrace to get cupti initialization to settle
  // std::cout << "Warm up " << std::endl;
  // warmup();
  // std::cout << "Start trace" << std::endl;
  // profiler.startTrace();
  // std::cout << "Start playground" << std::endl;

  for (int i = 0; i < 50; i++) {
    playground();
    // profiler.step();
    libkineto::api().activityProfiler().step();
    std::cout << "Step " << i << std::endl;
  }

  // std::cout << "Stop Trace" << std::endl;
  // auto trace = profiler.stopTrace();
  // std::cout << "Stopped and processed trace. Got "
  //           << trace->activities()->size() << " activities.";
  // trace->save(kFileName);
  return 0;
}
