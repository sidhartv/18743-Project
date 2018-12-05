/*
 * Copyright (c) 2005 The Regents of The University of Michigan
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: Ron Dreslinski
 */

/**
 * @file
 * Describes a tagged prefetcher.
 */

#ifndef __MEM_CACHE_PREFETCH_TAGGED_HH__
#define __MEM_CACHE_PREFETCH_TAGGED_HH__

// Embedded Python2.7 interpreter
#include <Python.h>

#include "mem/cache/prefetch/queued.hh"
#include "mem/packet.hh"

struct TaggedPrefetcherParams;

class TaggedPrefetcher : public QueuedPrefetcher {
 protected:
  static const char conda[];
  static const char modpath[];
  static const char modname[];
  static const char testfn_name[];
  static const char initfn_name[];
  static const char inferfn_name[];
  static const char inferchkfn_name[];
  static const char cleanupfn_name[];

  static const char rnn_archpath[];
  static const char rnn_weightspath[];
  static const char rnn_clusterpath[];

 protected:
  int degree;

  std::string pythonhome;
  std::string programname;

  PyObject *pModule;
  PyObject *pFn_test, *pFn_init, *pFn_inferchk, *pFn_infer, *pFn_cleanup;
  PyObject *pRNN_handle;

 public:
  TaggedPrefetcher(const TaggedPrefetcherParams *p);

  ~TaggedPrefetcher();

  void calculatePrefetch(const PacketPtr &pkt,
                         std::vector<AddrPriority> &addresses);
};

#endif  // __MEM_CACHE_PREFETCH_TAGGED_HH__
