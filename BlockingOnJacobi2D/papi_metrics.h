#pragma once

#include <papi.h>
#include <omp.h>
#include <cstdio>
#include <vector>
#include <cstring>

// Global PAPI event code
static int g_papi_event = PAPI_NULL;
static bool g_papi_enabled = false;

// Per-thread PAPI values
static std::vector<int> g_eventsets;
static std::vector<long long> g_papi_values;

static void init_papi(const char* metric_name) {
    if (!metric_name || strlen(metric_name) == 0) {
        g_papi_enabled = false;
        return;
    }

    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI library init error\n");
        return;
    }

    retval = PAPI_thread_init((unsigned long (*)(void))omp_get_thread_num);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI thread init error\n");
        return;
    }

    retval = PAPI_event_name_to_code((char*)metric_name, &g_papi_event);
    if (retval != PAPI_OK) {
        fprintf(stderr, "Event not recognized\n");
        return;
    }

    int nthreads = omp_get_max_threads();
    g_eventsets.resize(nthreads, PAPI_NULL);
    g_papi_values.resize(nthreads, 0);
    g_papi_enabled = true;
}

static void start_papi_thread() {
    if (!g_papi_enabled) return;

    int tid = omp_get_thread_num();
    int& EventSet = g_eventsets[tid];

    PAPI_register_thread();

    if (EventSet == PAPI_NULL) {
        PAPI_create_eventset(&EventSet);
        PAPI_add_event(EventSet, g_papi_event);
    }

    PAPI_start(EventSet);
}


void stop_papi_thread() {
    if (!g_papi_enabled) return;

    int tid = omp_get_thread_num();
    int EventSet = g_eventsets[tid];

    long long value = 0;
    PAPI_stop(EventSet, &value);

    g_papi_values[tid] = value;

    PAPI_unregister_thread();
}
