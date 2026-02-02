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
        fprintf(stderr, "PAPI library init error: %s\n", PAPI_strerror(retval));
        return;
    }

    retval = PAPI_thread_init((unsigned long (*)(void))omp_get_thread_num);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI thread init error: %s\n", PAPI_strerror(retval));
        return;
    }

    retval = PAPI_event_name_to_code((char*)metric_name, &g_papi_event);
    if (retval != PAPI_OK) {
        fprintf(stderr, "Event '%s' not recognized: %s\n", metric_name, PAPI_strerror(retval));
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
    if (tid < 0 || tid >= (int)g_eventsets.size()) {
        fprintf(stderr, "PAPI start error: thread %d out of range\n", tid);
        return;
    }

    int& EventSet = g_eventsets[tid];

    int retval = PAPI_register_thread();
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI register thread error (tid %d): %s\n", tid, PAPI_strerror(retval));
        return;
    }

    if (EventSet == PAPI_NULL) {
        retval = PAPI_create_eventset(&EventSet);
        if (retval != PAPI_OK) {
            fprintf(stderr, "PAPI create eventset error (tid %d): %s\n", tid, PAPI_strerror(retval));
            return;
        }

        retval = PAPI_add_event(EventSet, g_papi_event);
        if (retval != PAPI_OK) {
            fprintf(stderr, "PAPI add event error (tid %d): %s\n", tid, PAPI_strerror(retval));
            return;
        }
    }

    retval = PAPI_start(EventSet);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI start error (tid %d): %s\n", tid, PAPI_strerror(retval));
        return;
    }
}

static void stop_papi_thread() {
    if (!g_papi_enabled) return;

    int tid = omp_get_thread_num();
    if (tid < 0 || tid >= (int)g_eventsets.size()) {
        fprintf(stderr, "PAPI stop error: thread %d out of range\n", tid);
        return;
    }

    int EventSet = g_eventsets[tid];
    if (EventSet == PAPI_NULL) {
        fprintf(stderr, "PAPI stop error (tid %d): eventset not initialized\n", tid);
        return;
    }

    long long value = 0;
    int retval = PAPI_stop(EventSet, &value);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI stop error (tid %d): %s\n", tid, PAPI_strerror(retval));
        return;
    }

    g_papi_values[tid] = value;

    retval = PAPI_unregister_thread();
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI unregister thread error (tid %d): %s\n", tid, PAPI_strerror(retval));
    }
}

static void init_papi_single_thread(const char* metric_name) {
    if (!metric_name || strlen(metric_name) == 0) {
        g_papi_enabled = false;
        return;
    }

    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI library init error: %s\n", PAPI_strerror(retval));
        return;
    }

    retval = PAPI_event_name_to_code((char*)metric_name, &g_papi_event);
    if (retval != PAPI_OK) {
        fprintf(stderr, "Event '%s' not recognized: %s\n", metric_name, PAPI_strerror(retval));
        return;
    }

    g_papi_enabled = true;
    g_eventsets.resize(1, PAPI_NULL);
    g_papi_values.resize(1, 0);
}

static void start_papi_single_thread() {
    if (!g_papi_enabled) return;

    int retval;
    if (g_eventsets[0] == PAPI_NULL) {
        retval = PAPI_create_eventset(&g_eventsets[0]);
        if (retval != PAPI_OK) {
            fprintf(stderr, "PAPI create eventset error: %s\n", PAPI_strerror(retval));
            return;
        }

        retval = PAPI_add_event(g_eventsets[0], g_papi_event);
        if (retval != PAPI_OK) {
            fprintf(stderr, "PAPI add event error: %s\n", PAPI_strerror(retval));
            return;
        }
    }

    retval = PAPI_start(g_eventsets[0]);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI start error: %s\n", PAPI_strerror(retval));
        return;
    }
}

static void stop_papi_single_thread() {
    if (!g_papi_enabled) return;

    if (g_eventsets[0] == PAPI_NULL) {
        fprintf(stderr, "PAPI stop error: eventset not initialized\n");
        return;
    }

    int retval = PAPI_stop(g_eventsets[0], &g_papi_values[0]);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI stop error: %s\n", PAPI_strerror(retval));
    }
}

static long long get_papi_value() {
    if (!g_papi_enabled || g_papi_values.empty()) {
        return 0;
    }
    return g_papi_values[0];
}