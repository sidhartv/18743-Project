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

#include <wordexp.h>
#include <string>

// Static variable definitions
// const char TaggedPrefetcher::conda[]           = "${CONDA_PREFIX}/";
const char TaggedPrefetcher::conda[] =
    "/afs/ece/usr/dstiffle/miniconda3/envs/gem5/";
const char TaggedPrefetcher::modpath[]         = "${RNN_SCRIPTS}/";
const char TaggedPrefetcher::modname[]         = "RNN";
const char TaggedPrefetcher::testfn_name[]     = "test";
const char TaggedPrefetcher::initfn_name[]     = "init";
const char TaggedPrefetcher::inferfn_name[]    = "infer";
const char TaggedPrefetcher::inferchkfn_name[] = "inferchk";
const char TaggedPrefetcher::cleanupfn_name[]  = "cleanup";

const char TaggedPrefetcher::rnn_archpath[]    = "${RNN_PATH}/model.json";
const char TaggedPrefetcher::rnn_weightspath[] = "${RNN_PATH}/weights/";
const char TaggedPrefetcher::rnn_clusterpath[] = "${RNN_PATH}/cluster.csv";

TaggedPrefetcher::TaggedPrefetcher(const TaggedPrefetcherParams *p)
    : QueuedPrefetcher(p), degree(p->degree) {

  wordexp_t ex;

  {  // Program name must remain persistent
    DPRINTF(PyHook, "Setting up Python program name.\n");
    wordexp(conda, &ex, 0);
    for (int i = 0; i < ex.we_wordc; ++i) programname += ex.we_wordv[i];
    wordfree(&ex);
    programname += "bin/python";

    Py_SetProgramName(const_cast<char *>(programname.c_str()));

    const char *new_programname = Py_GetProgramName();
    DPRINTF(PyHook, "Program name is %s (should be %s).\n", new_programname,
            programname.c_str());
  }

  {  // PYTHONHOME must remain persistent
    DPRINTF(PyHook, "Setting up PYTHONHOME.\n");
    wordexp(conda, &ex, 0);
    for (int i = 0; i < ex.we_wordc; ++i) pythonhome += ex.we_wordv[i];
    wordfree(&ex);

    Py_SetPythonHome(const_cast<char *>(pythonhome.c_str()));

    const char *new_pythonhome = Py_GetPythonHome();
    DPRINTF(PyHook, "PYTHONHOME is %s (should be %s).\n", new_pythonhome,
            pythonhome.c_str());
  }

  Py_Initialize();  // Start the embedded interpreter
  if (!Py_IsInitialized()) {
    PyErr_Print();
    DPRINTF(PyHook, "ERR> Python environment was not initialized properly.\n");
    exit(EXIT_FAILURE);
  }

  {
    DPRINTF(PyHook, "Setting up PATH.\n");
    wordexp(modpath, &ex, 0);
    std::string expanded;
    for (int i = 0; i < ex.we_wordc; ++i) expanded += ex.we_wordv[i];
    wordfree(&ex);
    DPRINTF(PyHook, "PATH will have %s appended.\n", expanded.c_str());

    PyObject *py_sys_mod = PyImport_ImportModule("sys");
    PyObject *py_syspath = PyObject_GetAttrString(py_sys_mod, "path");
    PyList_Append(py_syspath, PyString_FromString(expanded.c_str()));
    Py_DECREF(py_syspath);
    Py_DECREF(py_sys_mod);

    DPRINTF(PyHook,
            "PATH is %s. prefix is %s. exec-prefix is %s, full path is %s.\n",
            Py_GetPath(), Py_GetPrefix(), Py_GetExecPrefix(),
            Py_GetProgramFullPath());
  }

  DPRINTF(PyHook, "Attempting to import module -%s-\n", modname);
  pModule = PyImport_ImportModule(modname);
  if (pModule == nullptr) {
    PyErr_Print();
    DPRINTF(PyHook, "Err> Unable to import module -%s- @(%p)\n", modname);

    exit(EXIT_FAILURE);
  }

  DPRINTF(PyHook, "Successfully imported module -%s- @(%p)\n", modname,
          (void *)pModule);

  pFn_init = PyObject_GetAttrString(pModule, initfn_name);
  if ((pFn_init == nullptr) || !PyCallable_Check(pFn_init)) {
    PyErr_Print();
    DPRINTF(PyHook, "ERR> Unable to (init) register function hook -%s.%s-.\n",
            modname, testfn_name);

    exit(EXIT_FAILURE);
  }

  pFn_inferchk = PyObject_GetAttrString(pModule, inferchkfn_name);
  if ((pFn_inferchk == nullptr) || !PyCallable_Check(pFn_inferchk)) {
    PyErr_Print();
    DPRINTF(PyHook,
            "ERR> Unable to (inferchk) register function hook -%s.%s-.\n",
            modname, testfn_name);

    exit(EXIT_FAILURE);
  }

  pFn_infer = PyObject_GetAttrString(pModule, inferfn_name);
  if ((pFn_infer == nullptr) || !PyCallable_Check(pFn_infer)) {
    PyErr_Print();
    DPRINTF(PyHook, "ERR> Unable to (infer) register function hook -%s.%s-.\n",
            modname, testfn_name);

    exit(EXIT_FAILURE);
  }

  pFn_cleanup = PyObject_GetAttrString(pModule, cleanupfn_name);
  if ((pFn_cleanup == nullptr) || !PyCallable_Check(pFn_cleanup)) {
    PyErr_Print();
    DPRINTF(PyHook,
            "ERR> Unable to (cleanup) register function hook -%s.%s-.\n",
            modname, testfn_name);

    exit(EXIT_FAILURE);
  }

  DPRINTF(PyHook,
          "Successfully registered function hooks init@(%p) inferchk@(%p) "
          "infer@(%p) cleanup@(%p).\n",
          (void *)pFn_init, (void *)pFn_inferchk, (void *)pFn_infer,
          (void *)pFn_cleanup);

  PyObject *init_args =
      Py_BuildValue("(s s s)", rnn_archpath, rnn_weightspath, rnn_clusterpath);
  pRNN_handle = PyObject_CallObject(pFn_init, init_args);
  if (pRNN_handle == nullptr) {
    DPRINTF(PyHook, "Unable to interpret output from -%s.%s-.\n", modname,
            initfn_name);

    exit(EXIT_FAILURE);
  }
  Py_DECREF(init_args);

  DPRINTF(PyHook,
          "Successfully registered hooks to module RNN@(%p) from -%s.%s-.\n",
          (void *)pRNN_handle, modname, initfn_name);
}

TaggedPrefetcher::~TaggedPrefetcher() {
  (void *)PyObject_CallObject(pFn_cleanup, nullptr);

  Py_DECREF(pRNN_handle);

  Py_DECREF(pFn_init);
  Py_DECREF(pFn_inferchk);
  Py_DECREF(pFn_infer);
  Py_DECREF(pFn_cleanup);

  Py_DECREF(pModule);

  Py_Finalize();
}

void TaggedPrefetcher::calculatePrefetch(const PacketPtr &pkt,
                                         std::vector<AddrPriority> &addresses) {

  Addr daddr   = pkt->getBlockAddr(blkSize);
  Addr iaddr   = pkt->req->getPC();
  bool is_read = pkt->isRead();
  /*
  DPRINTF(PyHook, "daddr(0x%0llx) iaddr(0x%0llx) is_read(%d)\n", daddr, iaddr,
          is_read);
  */

  PyObject *infer_args = Py_BuildValue("(K K N O)", iaddr, daddr,
                                       PyBool_FromLong(is_read), pRNN_handle);

  PyObject *ret;

  ret = PyObject_CallObject(pFn_inferchk, infer_args);
  if (ret == nullptr) {
    PyErr_Print();
    DPRINTF(PyHook,
            "ERR> Unable to determine inference validity: infer_args(%p).\n",
            (void *)infer_args);

    exit(EXIT_FAILURE);
  }

  bool is_valid = PyObject_IsTrue(ret);
  if (!is_valid) return;

  ret = PyObject_CallObject(pFn_infer, infer_args);
  if (ret == nullptr) {
    PyErr_Print();
    DPRINTF(PyHook, "Received nullptr from call to -%s.%s-.\n", modname,
            inferfn_name);

    return;
  }
  Py_DECREF(infer_args);

  int infer_len = PyList_Size(ret);
  DPRINTF(PyHook, "PF> Handling %d inferenced prefetchers.\n", infer_len);

  for (int i = 0; i < infer_len; ++i) {
    PyObject *tmp              = PyList_GetItem(ret, i);
    unsigned long long tmp_ull = -1;
    if (PyLong_Check(tmp)) {
      tmp_ull = PyLong_AsUnsignedLongLong(tmp);
    } else if (PyInt_Check(tmp)) {
      tmp_ull = PyInt_AsUnsignedLongLongMask(tmp);
    }
    Addr pref_addr = static_cast<Addr>(tmp_ull);

    DPRINTF(PyHook,
            "PF> Issuing request for daddr(0x%0llx)->pref_addr(0x%0llx).\n",
            daddr, pref_addr);

    if (!samePage(daddr, pref_addr)) {
      // Count number of unissued prefetches due to page crossing
      pfSpanPage += infer_len - i + 1;

      if (i == 1) {
        pfSingle += 1;
      } else if (i > 1) {
        pfMultiple += 1;
      }

      Py_DECREF(ret);

      return;
    } else {
      addresses.push_back(AddrPriority(pref_addr, 0));
    }
  }

  if (infer_len == 1) {
    pfSingle += 1;
  } else if (infer_len > 1) {
    pfMultiple += 1;
  }

  Py_DECREF(ret);
}

TaggedPrefetcher *TaggedPrefetcherParams::create() {
  return new TaggedPrefetcher(this);
}
