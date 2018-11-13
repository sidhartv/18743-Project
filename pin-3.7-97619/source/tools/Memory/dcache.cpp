/*BEGIN_LEGAL
Intel Open Source License

Copyright (c) 2002-2018 Intel Corporation. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.  Redistributions
in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.  Neither the name of
the Intel Corporation nor the names of its contributors may be used to
endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE INTEL OR
ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
END_LEGAL */
/*! @file
 *  This file contains an ISA-portable cache simulator
 *  data cache hierarchies
 */

#include "pin.H"

#include <cassert>
#include <fstream>
#include <iostream>

#include "cache.H"
#include "pin_profile.H"

/* ===================================================================== */
/* Helper POD struct definitions */
/* ===================================================================== */

struct atrace {
  void* iaddr;
  void* daddr;
  bool is_read;
};

/* ===================================================================== */
/* Commandline Switches */
/* ===================================================================== */

KNOB<string> KnobOutputFile(KNOB_MODE_WRITEONCE, "pintool", "o", "dcache.out",
                            "specify dcache file name");
KNOB<string> KnobTraceFile(KNOB_MODE_WRITEONCE, "pintool", "ot", "atrace.out",
                           "specify access trace file name");
KNOB<BOOL> KnobTrackLoads(KNOB_MODE_WRITEONCE, "pintool", "tl", "0",
                          "track individual loads -- increases profiling time");
KNOB<BOOL> KnobTrackStores(
    KNOB_MODE_WRITEONCE, "pintool", "ts", "0",
    "track individual stores -- increases profiling time");
KNOB<UINT32> KnobThresholdHit(
    KNOB_MODE_WRITEONCE, "pintool", "rh", "100",
    "only report memops with hit count above threshold");
KNOB<UINT32> KnobThresholdMiss(
    KNOB_MODE_WRITEONCE, "pintool", "rm", "100",
    "only report memops with miss count above threshold");
KNOB<UINT32> KnobCacheSize(KNOB_MODE_WRITEONCE, "pintool", "c", "32",
                           "cache size in kilobytes");
KNOB<UINT32> KnobLineSize(KNOB_MODE_WRITEONCE, "pintool", "b", "64",
                          "cache block size in bytes");
KNOB<UINT32> KnobAssociativity(KNOB_MODE_WRITEONCE, "pintool", "a", "4",
                               "cache associativity (1 for direct mapped)");

/* ===================================================================== */

INT32 Usage() {
  cerr << "This tool represents a cache simulator.\n"
          "\n";

  cerr << KNOB_BASE::StringKnobSummary();

  cerr << endl;

  return -1;
}

/* ===================================================================== */
/* Global Variables */

FILE* trace_stream;

/* ===================================================================== */

// wrap configuation constants into their own name space to avoid name clashes
namespace DL1 {
const UINT32 max_sets = KILO;  // cacheSize / (lineSize * associativity);
const UINT32 max_associativity                 = 256;  // associativity;
const CACHE_ALLOC::STORE_ALLOCATION allocation = CACHE_ALLOC::STORE_ALLOCATE;

typedef CACHE_ROUND_ROBIN(max_sets, max_associativity, allocation) CACHE;
}

DL1::CACHE* dl1 = NULL;

typedef enum { COUNTER_MISS = 0, COUNTER_HIT = 1, COUNTER_NUM } COUNTER;

typedef COUNTER_ARRAY<UINT64, COUNTER_NUM> COUNTER_HIT_MISS;

// holds the counters with misses and hits
// conceptually this is an array indexed by instruction address
COMPRESSOR_COUNTER<ADDRINT, UINT32, COUNTER_HIT_MISS> profile;

// holds a mapping between instruction addresses and individual
// COMPRESSOR_COUNTER objects for the data addresses
typedef COMPRESSOR_COUNTER<ADDRINT, UINT32, COUNTER_HIT_MISS>
    INDIRECT_COMPRESSOR_COUNTER;
std::map<ADDRINT, INDIRECT_COMPRESSOR_COUNTER> indirect_profile;
typedef std::map<ADDRINT, INDIRECT_COMPRESSOR_COUNTER>::const_iterator
    indirect_profile_it_t;

/* ===================================================================== */

VOID LoadMulti(ADDRINT addr, UINT32 size, UINT32 instId, ADDRINT iaddr) {
  // first level D-cache
  const BOOL dl1Hit = dl1->Access(addr, size, CACHE_BASE::ACCESS_TYPE_LOAD);

  const COUNTER counter = dl1Hit ? COUNTER_HIT : COUNTER_MISS;
  profile[instId][counter]++;

  // Map indirection from instruction to data address and increment the
  // appropriate counter based on the cache access
  if (indirect_profile.find(iaddr) == indirect_profile.end()) {
    COMPRESSOR_COUNTER<ADDRINT, UINT32, COUNTER_HIT_MISS> new_compressor(64);
    new_compressor.SetKeyName("daddr          ");
    new_compressor.SetCounterName("dcache:miss        dcache:hit");

    COUNTER_HIT_MISS threshold;
    threshold[COUNTER_HIT]  = KnobThresholdHit.Value();
    threshold[COUNTER_MISS] = 1;
    new_compressor.SetThreshold(threshold);

    indirect_profile.insert(
        std::pair<ADDRINT, INDIRECT_COMPRESSOR_COUNTER>(iaddr, new_compressor));
  }

  UINT32 memId = indirect_profile[iaddr].Map(addr);
  indirect_profile[iaddr][memId][counter]++;
}

/* ===================================================================== */

VOID StoreMulti(ADDRINT addr, UINT32 size, UINT32 instId, ADDRINT iaddr) {
  // first level D-cache
  const BOOL dl1Hit = dl1->Access(addr, size, CACHE_BASE::ACCESS_TYPE_STORE);

  const COUNTER counter = dl1Hit ? COUNTER_HIT : COUNTER_MISS;
  profile[instId][counter]++;

  // Map indirection from instruction to data address and increment the
  // appropriate counter based on the cache access
  if (indirect_profile.find(iaddr) == indirect_profile.end()) {
    COMPRESSOR_COUNTER<ADDRINT, UINT32, COUNTER_HIT_MISS> new_compressor(64);
    new_compressor.SetKeyName("daddr          ");
    new_compressor.SetCounterName("dcache:miss        dcache:hit");

    COUNTER_HIT_MISS threshold;
    threshold[COUNTER_HIT]  = KnobThresholdHit.Value();
    threshold[COUNTER_MISS] = 1;
    new_compressor.SetThreshold(threshold);

    indirect_profile.insert(
        std::pair<ADDRINT, INDIRECT_COMPRESSOR_COUNTER>(iaddr, new_compressor));
  }

  UINT32 memId = indirect_profile[iaddr].Map(addr);
  indirect_profile[iaddr][memId][counter]++;
}

/* ===================================================================== */

VOID LoadSingle(ADDRINT addr, UINT32 instId, ADDRINT iaddr) {
  // @todo we may access several cache lines for
  // first level D-cache
  const BOOL dl1Hit = dl1->AccessSingleLine(addr, CACHE_BASE::ACCESS_TYPE_LOAD);

  const COUNTER counter = dl1Hit ? COUNTER_HIT : COUNTER_MISS;
  profile[instId][counter]++;

  // Map indirection from instruction to data address and increment the
  // appropriate counter based on the cache access
  if (indirect_profile.find(iaddr) == indirect_profile.end()) {
    COMPRESSOR_COUNTER<ADDRINT, UINT32, COUNTER_HIT_MISS> new_compressor(64);
    new_compressor.SetKeyName("daddr          ");
    new_compressor.SetCounterName("dcache:miss        dcache:hit");

    COUNTER_HIT_MISS threshold;
    threshold[COUNTER_HIT]  = KnobThresholdHit.Value();
    threshold[COUNTER_MISS] = 1;
    new_compressor.SetThreshold(threshold);

    indirect_profile.insert(
        std::pair<ADDRINT, INDIRECT_COMPRESSOR_COUNTER>(iaddr, new_compressor));
  }

  UINT32 memId = indirect_profile[iaddr].Map(addr);
  indirect_profile[iaddr][memId][counter]++;
}
/* ===================================================================== */

VOID StoreSingle(ADDRINT addr, UINT32 instId, ADDRINT iaddr) {
  // @todo we may access several cache lines for
  // first level D-cache
  const BOOL dl1Hit =
      dl1->AccessSingleLine(addr, CACHE_BASE::ACCESS_TYPE_STORE);

  const COUNTER counter = dl1Hit ? COUNTER_HIT : COUNTER_MISS;
  profile[instId][counter]++;

  // Map indirection from instruction to data address and increment the
  // appropriate counter based on the cache access
  if (indirect_profile.find(iaddr) == indirect_profile.end()) {
    COMPRESSOR_COUNTER<ADDRINT, UINT32, COUNTER_HIT_MISS> new_compressor(64);
    new_compressor.SetKeyName("daddr          ");
    new_compressor.SetCounterName("dcache:miss        dcache:hit");

    COUNTER_HIT_MISS threshold;
    threshold[COUNTER_HIT]  = KnobThresholdHit.Value();
    threshold[COUNTER_MISS] = 1;
    new_compressor.SetThreshold(threshold);

    indirect_profile.insert(
        std::pair<ADDRINT, INDIRECT_COMPRESSOR_COUNTER>(iaddr, new_compressor));
  }

  UINT32 memId = indirect_profile[iaddr].Map(addr);
  indirect_profile[iaddr][memId][counter]++;
}

/* ===================================================================== */

VOID LoadMultiFast(ADDRINT addr, UINT32 size) {
  dl1->Access(addr, size, CACHE_BASE::ACCESS_TYPE_LOAD);
}

/* ===================================================================== */

VOID StoreMultiFast(ADDRINT addr, UINT32 size) {
  dl1->Access(addr, size, CACHE_BASE::ACCESS_TYPE_STORE);
}

/* ===================================================================== */

VOID LoadSingleFast(ADDRINT addr) {
  dl1->AccessSingleLine(addr, CACHE_BASE::ACCESS_TYPE_LOAD);
}

/* ===================================================================== */

VOID StoreSingleFast(ADDRINT addr) {
  dl1->AccessSingleLine(addr, CACHE_BASE::ACCESS_TYPE_STORE);
}

// Print a memory read record
VOID RecordMemRead(VOID* ip, VOID* addr) {
  struct atrace tmp = {ip, addr, true};
  size_t nmemb      = sizeof(tmp);

  size_t write_retval = fwrite(&tmp, nmemb, 1, trace_stream);
  if (write_retval != 1) {
    int err = ferror(trace_stream);
    std::cerr << "-> " << strerror(err) << ".\n";

    exit(EXIT_FAILURE);
  }
}

// Print a memory write record
VOID RecordMemWrite(VOID* ip, VOID* addr) {
  struct atrace tmp = {ip, addr, false};
  size_t nmemb      = sizeof(tmp);

  size_t write_retval = fwrite(&tmp, nmemb, 1, trace_stream);
  if (write_retval != 1) {
    int err = ferror(trace_stream);
    std::cerr << "-> " << strerror(err) << ".\n";

    exit(EXIT_FAILURE);
  }
}

/* ===================================================================== */

VOID Instruction(INS ins, void* v) {
  UINT32 memOperands = INS_MemoryOperandCount(ins);

  // Instrument each memory operand. If the operand is both read and written
  // it will be processed twice.
  // Iterating over memory operands ensures that instructions on IA-32 with
  // two read operands (such as SCAS and CMPS) are correctly handled.
  for (UINT32 memOp = 0; memOp < memOperands; memOp++) {
    const UINT32 size = INS_MemoryOperandSize(ins, memOp);
    const BOOL single = (size <= 4);

    if (INS_MemoryOperandIsRead(ins, memOp)) {
      if (KnobTrackLoads) {
        INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR)RecordMemRead,
                                 IARG_INST_PTR, IARG_MEMORYOP_EA, memOp,
                                 IARG_END);

        // map sparse INS addresses to dense IDs
        const ADDRINT iaddr = INS_Address(ins);
        const UINT32 instId = profile.Map(iaddr);

        if (single) {
          INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR)LoadSingle,
                                   IARG_MEMORYOP_EA, memOp, IARG_UINT32, instId,
                                   IARG_ADDRINT, iaddr, IARG_END);
        } else {
          INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR)LoadMulti,
                                   IARG_MEMORYOP_EA, memOp, IARG_UINT32, size,
                                   IARG_UINT32, instId, IARG_ADDRINT, iaddr,
                                   IARG_END);
        }
      } else {
        if (single) {
          INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR)LoadSingleFast,
                                   IARG_MEMORYOP_EA, memOp, IARG_END);

        } else {
          INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR)LoadMultiFast,
                                   IARG_MEMORYOP_EA, memOp, IARG_UINT32, size,
                                   IARG_END);
        }
      }
    }

    if (INS_MemoryOperandIsWritten(ins, memOp)) {
      if (KnobTrackStores) {
        INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR)RecordMemWrite,
                                 IARG_INST_PTR, IARG_MEMORYOP_EA, memOp,
                                 IARG_END);

        const ADDRINT iaddr = INS_Address(ins);
        const UINT32 instId = profile.Map(iaddr);

        if (single) {
          INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR)StoreSingle,
                                   IARG_MEMORYOP_EA, memOp, IARG_UINT32, instId,
                                   IARG_ADDRINT, iaddr, IARG_END);
        } else {
          INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR)StoreMulti,
                                   IARG_MEMORYOP_EA, memOp, IARG_UINT32, size,
                                   IARG_UINT32, instId, IARG_ADDRINT, iaddr,
                                   IARG_END);
        }
      } else {
        if (single) {
          INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR)StoreSingleFast,
                                   IARG_MEMORYOP_EA, memOp, IARG_END);

        } else {
          INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR)StoreMultiFast,
                                   IARG_MEMORYOP_EA, memOp, IARG_UINT32, size,
                                   IARG_END);
        }
      }
    }
  }
}

/* ===================================================================== */

VOID Fini(int code, VOID* v) {

  std::ofstream out(KnobOutputFile.Value().c_str());

  // print D-cache profile
  // @todo what does this print

  out << "PIN:MEMLATENCIES 1.0. 0x0\n";

  out << "#\n"
         "# DCACHE stats\n"
         "#\n";

  out << dl1->StatsLong("# ", CACHE_BASE::CACHE_TYPE_DCACHE);

  if (KnobTrackLoads || KnobTrackStores) {
    out << "#\n"
           "# Begin LOAD/STORE stats\n"
           "#\n";

    out << profile.StringLong();

    for (indirect_profile_it_t it = indirect_profile.begin();
         it != indirect_profile.end(); it++) {

      out << "#\n# Detailed Stats\n# Instruction Address: "
          << hexstr(it->first, 8) << "\n"
          << it->second.StringLong();
    }

    out << "#\n"
           "# End LOAD/STORE stats\n"
           "#\n";
  }
  out.close();

  fclose(trace_stream);
}

/* ===================================================================== */

int main(int argc, char* argv[]) {
  PIN_InitSymbols();

  if (PIN_Init(argc, argv)) {
    return Usage();
  }

  dl1 = new DL1::CACHE("L1 Data Cache", KnobCacheSize.Value() * KILO,
                       KnobLineSize.Value(), KnobAssociativity.Value());

  profile.SetKeyName("iaddr          ");
  profile.SetCounterName("dcache:miss        dcache:hit");

  COUNTER_HIT_MISS threshold;

  threshold[COUNTER_HIT]  = KnobThresholdHit.Value();
  threshold[COUNTER_MISS] = KnobThresholdMiss.Value();

  profile.SetThreshold(threshold);

  trace_stream = fopen(KnobTraceFile.Value().c_str(), "wb");
  if (trace_stream == NULL) {
    std::cerr << "Unable to open " << KnobTraceFile.Value().c_str()
              << " for binary writing.\n";

    exit(EXIT_FAILURE);
  }

  INS_AddInstrumentFunction(Instruction, 0);
  PIN_AddFiniFunction(Fini, 0);

  // Never returns

  PIN_StartProgram();

  return 0;
}

/* ===================================================================== */
/* eof */
/* ===================================================================== */
