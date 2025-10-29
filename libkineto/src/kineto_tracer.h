#pragma once

#include <arrow/type_fwd.h>
#include <mon_client/mpi_client.h>
#include <mon_client/schema_tracer.h>
#include <mon_client/trace_types.h>
#include <mon_client/tracer_macros.h>
#include <cstdint>

namespace KINETO_NAMESPACE {
using EventBase = mon::client::EventBase;
template <typename T>
using SchemaTracer = mon::client::SchemaTracer<T>;

struct KinetoEvent {
  EventBase base;
  std::string cat;
  std::string name;
  int64_t pid;
  int64_t tid;
  int64_t ts;
  int64_t dur;
  int64_t end;
  int64_t stream;
  int64_t correlation;
  int64_t bytes;
  double mem_bw;
  int64_t wait_on_stream;
  int64_t wait_on_cuda_event;
};

#define KINETO_FIELDS(F)                                  \
  F(Int32, rank, event.base.rank, int32())                \
  F(String, cat, event.cat, utf8())                       \
  F(String, name, event.name, utf8())                     \
  F(Int64, pid, event.pid, int64())                       \
  F(Int64, tid, event.tid, int64())                       \
  F(Int64, ts, event.ts, int64())                         \
  F(Int64, dur, event.dur, int64())                       \
  F(Int64, end, event.end, int64())                       \
  F(Int64, stream, event.stream, int64())                 \
  F(Int64, correlation, event.correlation, int64())       \
  F(Int64, bytes, event.bytes, int64())                   \
  F(Double, mem_bw, event.mem_bw, float64())              \
  F(Int64, wait_on_stream, event.wait_on_stream, int64()) \
  F(Int64, wait_on_cuda_event, event.wait_on_cuda_event, int64())

class KinetoTracer : public SchemaTracer<KinetoEvent> {
 public:
  KinetoTracer(const char* schema_name, bool debug_mode = false)
      : SchemaTracer<KinetoEvent>(schema_name, debug_mode)
            INIT_FIELDS(KINETO_FIELDS) {}

  GENERATE_TRACER_METHODS(KinetoEvent, KINETO_FIELDS)
};

using KinetoTracerRef = std::shared_ptr<KinetoTracer>;

} // namespace KINETO_NAMESPACE