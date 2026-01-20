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

// Dedicated event type for torch ops as they are dominant in the kineto trace
struct KinetoTorchOpEvent {
  EventBase base;
  int8_t ph; // a single char, use int8_t to represent
  std::string cat;
  std::string name;
  int64_t pid;
  int64_t tid;
  float ts;
  float dur;
  // appears for each torch op but never used, promote it to save space
  // otherwise saving it in args as string would waste space
  int64_t external_id;
  int64_t stream; // promoted for HTA, not all torch ops have this field
  int64_t correlation; // promoted for HTA, not all torch ops have this field
  float mem_bw_gbps; // promoted for HTA, not all torch ops have this field

  // encapsulate the various fields in the "args" field into a single string
  // in the format of "key1=value1,key2=value2,..."
  std::string args;
};

// Include the remaining types of events in the kineto trace, e.g. span,
// device_info, links, etc. Some of them may have unique fields but since they
// appear rarely we put them in args
struct KinetoMiscEvent {
  EventBase base;
  int8_t ph;
  std::string cat;
  std::string name;
  int64_t pid;
  int64_t tid;
  float ts;
  std::string args;
};

// Used to encode arbitrary metadata in key-value pairs
struct KinetoMetadataEvent {
  EventBase base;
  std::string key;
  std::string value;
};

#define TORCH_OP_EVENT_FIELDS(F)                      \
  F(Int32, timestep, event.base.timestep, int32())    \
  F(UInt64, swid, event.base.swid, uint64())          \
  F(Int32, rank, event.base.rank, int32())            \
  F(UInt64, ts_ns, event.base.ts_ns, uint64())        \
  F(Int8, ph, event.ph, int8())                       \
  F(STRDICT_BUILDER, cat, event.cat, STRDICT_TYPE)    \
  F(STRDICT_BUILDER, name, event.name, STRDICT_TYPE)  \
  F(Int64, pid, event.pid, int64())                   \
  F(Int64, tid, event.tid, int64())                   \
  F(Float, ts, event.ts, float32())                   \
  F(Float, dur, event.dur, float32())                 \
  F(Int64, external_id, event.external_id, int64())   \
  F(Int64, stream, event.stream, int64())             \
  F(Int64, correlation, event.correlation, int64())   \
  F(Float, mem_bw_gbps, event.mem_bw_gbps, float32()) \
  F(String, args, event.args, utf8())

#define MISC_EVENT_FIELDS(F)                         \
  F(Int32, timestep, event.base.timestep, int32())   \
  F(UInt64, swid, event.base.swid, uint64())         \
  F(Int32, rank, event.base.rank, int32())           \
  F(UInt64, ts_ns, event.base.ts_ns, uint64())       \
  F(Int8, ph, event.ph, int8())                      \
  F(STRDICT_BUILDER, cat, event.cat, STRDICT_TYPE)   \
  F(STRDICT_BUILDER, name, event.name, STRDICT_TYPE) \
  F(Int64, pid, event.pid, int64())                  \
  F(Int64, tid, event.tid, int64())                  \
  F(Float, ts, event.ts, float32())                  \
  F(String, args, event.args, utf8())

#define METADATA_EVENT_FIELDS(F)                   \
  F(Int32, timestep, event.base.timestep, int32()) \
  F(UInt64, swid, event.base.swid, uint64())       \
  F(Int32, rank, event.base.rank, int32())         \
  F(UInt64, ts_ns, event.base.ts_ns, uint64())     \
  F(String, key, event.key, utf8())                \
  F(String, value, event.value, utf8())

class KinetoTorchOpTracer : public SchemaTracer<KinetoTorchOpEvent> {
 public:
  KinetoTorchOpTracer(const char* schema_name, bool debug_mode = false)
      : SchemaTracer<KinetoTorchOpEvent>(schema_name, debug_mode)
            INIT_FIELDS(TORCH_OP_EVENT_FIELDS) {}

  GENERATE_TRACER_METHODS(KinetoTorchOpEvent, TORCH_OP_EVENT_FIELDS)
};

class KinetoMiscTracer : public SchemaTracer<KinetoMiscEvent> {
 public:
  KinetoMiscTracer(const char* schema_name, bool debug_mode = false)
      : SchemaTracer<KinetoMiscEvent>(schema_name, debug_mode)
            INIT_FIELDS(MISC_EVENT_FIELDS) {}

  GENERATE_TRACER_METHODS(KinetoMiscEvent, MISC_EVENT_FIELDS)
};

class KinetoMetadataTracer : public SchemaTracer<KinetoMetadataEvent> {
 public:
  KinetoMetadataTracer(const char* schema_name, bool debug_mode = false)
      : SchemaTracer<KinetoMetadataEvent>(schema_name, debug_mode)
            INIT_FIELDS(METADATA_EVENT_FIELDS) {}

  GENERATE_TRACER_METHODS(KinetoMetadataEvent, METADATA_EVENT_FIELDS)
};

constexpr static const char* kKinetoTorchOpTracerSchema =
    "kineto_torch_op_events";
constexpr static const char* kKinetoMiscTracerSchema = "kineto_misc_events";
constexpr static const char* kKinetoMetadataTracerSchema =
    "kineto_metadata_events";

using KinetoTorchOpTracerRef = std::shared_ptr<KinetoTorchOpTracer>;
using KinetoMiscTracerRef = std::shared_ptr<KinetoMiscTracer>;
using KinetoMetadataTracerRef = std::shared_ptr<KinetoMetadataTracer>;
} // namespace KINETO_NAMESPACE