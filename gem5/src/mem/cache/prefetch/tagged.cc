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
 * Describes a tagged prefetcher based on template policies.
 */

#include "mem/cache/prefetch/tagged.hh"

#include "params/TaggedPrefetcher.hh"

#include "debug/PyHook.hh"

// Static variable definitions
const char TaggedPrefetcher::modpath[] = \
          "/afs/ece/usr/dstiffle/Private/ece/18743-Project/scripts/";
const char TaggedPrefetcher::modname[] = "testmodule";
const char TaggedPrefetcher::initfn_name[] = "init";
const char TaggedPrefetcher::testfn_name[] = "test";
const char TaggedPrefetcher::inferfn_name[] = "infer";

TaggedPrefetcher::TaggedPrefetcher(const TaggedPrefetcherParams *p)
    : QueuedPrefetcher(p), degree(p->degree)
{
  /* Py_SetProgramName(*name*);  // NOTE: this is optional but recommended */
  Py_Initialize();  // Start the embedded interpreter
  if (!Py_IsInitialized()) {
    PyErr_Print();
    DPRINTF(PyHook, "ERR> Python environment was not initialized properly.\n");
    exit(EXIT_FAILURE);
  }

  DPRINTF(PyHook, "Setting up PYTHONPATH.\n");
  PyObject *py_sys_mod = PyImport_ImportModule("sys");
  PyObject *py_syspath = PyObject_GetAttrString(py_sys_mod, "path");
  PyList_Append(py_syspath, PyString_FromString(modpath));
  Py_DECREF(py_syspath);
  Py_DECREF(py_sys_mod);

  DPRINTF(PyHook, "Attempting to import module -%s-\n", modname);
  pModule = PyImport_ImportModule(modname);
  if (pModule != nullptr) {
    DPRINTF(PyHook, "Successfully imported module -%s- @(%p)\n",
            modname, (void *) pModule);

    pFn_test = PyObject_GetAttrString(pModule, testfn_name);
    if ((pFn_test != nullptr) && PyCallable_Check(pFn_test)) {
      DPRINTF(PyHook, "Successfully registered function hook -%s.%s- @(%p)\n",
              modname, testfn_name, (void *) pFn_test);

      PyObject *pArgs = PyTuple_New(2);
      PyTuple_SetItem(pArgs, 0, PyInt_FromLong(18743));
      PyTuple_SetItem(pArgs, 1, PyInt_FromLong(0xdeadbeef));

      DPRINTF(PyHook, "Successfully built function arguments @(%p)\n",
              (void *) pArgs);

      PyObject *ret = PyObject_CallObject(pFn_test, pArgs);
      DPRINTF(PyHook, "Result of calling -%s.%s-: %d\n",
              modname, testfn_name, PyInt_AsLong(ret));
      Py_DECREF(pArgs);
      Py_DECREF(ret);
    }
  } else {
    PyErr_Print();
    DPRINTF(PyHook, "ERR> Unable to import module.\n");
    exit(EXIT_FAILURE);
  }
}

TaggedPrefetcher::~TaggedPrefetcher() {
  if (pModule != nullptr) {
    Py_DECREF(pFn_init);
    Py_DECREF(pFn_infer);

    Py_DECREF(pModule);
  }

  Py_Finalize();
}

void
TaggedPrefetcher::calculatePrefetch(const PacketPtr &pkt,
        std::vector<AddrPriority> &addresses)
{
    Addr blkAddr = pkt->getAddr() & ~(Addr)(blkSize-1);

    for (int d = 1; d <= degree; d++) {
        Addr newAddr = blkAddr + d*(blkSize);
        if (!samePage(blkAddr, newAddr)) {
            // Count number of unissued prefetches due to page crossing
            pfSpanPage += degree - d + 1;
            return;
        } else {
            addresses.push_back(AddrPriority(newAddr,0));
        }
    }
}

TaggedPrefetcher*
TaggedPrefetcherParams::create()
{
   return new TaggedPrefetcher(this);
}
