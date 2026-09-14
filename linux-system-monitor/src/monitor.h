/*
 * monitor.h - Main header file for Linux System Monitor
 *
 * This header is included by all source files. It defines:
 *   - Shared constants (e.g., buffer sizes, limits)
 *   - Shared data structures (e.g., CpuStats, MemInfo, ProcessInfo)
 *   - Function declarations (prototypes) for every module
 *
 * Linux concept: In C, header files let us share declarations across
 * multiple .c files without duplicating code. The compiler uses them
 * to check that function calls match their definitions.
 */

/*
 * Feature-test macro: must appear BEFORE any #include.
 * _DEFAULT_SOURCE exposes POSIX.1-2008 extensions (usleep, useconds_t,
 * strdup, etc.) even when compiling with -std=c11.
 * Without this, GCC hides non-C11 symbols and usleep() appears undeclared.
 */
#define _DEFAULT_SOURCE

#ifndef MONITOR_H   /* Include guard: prevents this header from being   */
#define MONITOR_H   /* processed twice in the same compilation unit.    */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>     /* POSIX: getpid(), sleep(), usleep()           */
#include <signal.h>     /* POSIX: kill(), SIGTERM, signal names         */
#include <dirent.h>     /* POSIX: opendir(), readdir(), closedir()      */
#include <ctype.h>      /* isdigit()                                    */
#include <errno.h>      /* errno variable and error codes               */
#include <sys/types.h>  /* pid_t, uid_t, etc.                           */
#include <sys/statvfs.h>/* statvfs() for disk usage                     */

/* ---------------------------------------------------------------
 * Constants
 * --------------------------------------------------------------- */

#define MAX_PROCESSES       512   /* Maximum number of processes we track  */
#define MAX_NAME_LEN        256   /* Maximum length for a process name      */
#define PROC_PATH_LEN       64    /* Length of a /proc/[pid]/... path       */
#define LINE_BUF_SIZE       512   /* General-purpose line read buffer       */
#define MONITOR_REFRESH_MS  1000  /* Live monitor refresh interval (ms)     */

/* ---------------------------------------------------------------
 * Data Structures
 * --------------------------------------------------------------- */

/*
 * CpuStats - raw counters read from /proc/stat
 *
 * /proc/stat exposes cumulative CPU time in "jiffies" (clock ticks).
 * To calculate a percentage, we read the values TWICE with a short
 * delay and compare the difference.
 *
 * Fields (all from the first "cpu" line of /proc/stat):
 *   user    - time in user mode
 *   nice    - time in user mode with low priority (niced)
 *   system  - time in kernel mode
 *   idle    - time doing nothing
 *   iowait  - time waiting for I/O to complete
 *   irq     - time servicing hardware interrupts
 *   softirq - time servicing software interrupts
 */
typedef struct {
    long long user;
    long long nice;
    long long system;
    long long idle;
    long long iowait;
    long long irq;
    long long softirq;
} CpuStats;

/*
 * MemInfo - memory figures read from /proc/meminfo (in kilobytes)
 */
typedef struct {
    long long total_kb;     /* MemTotal  */
    long long free_kb;      /* MemFree   */
    long long available_kb; /* MemAvailable */
    long long buffers_kb;   /* Buffers   */
    long long cached_kb;    /* Cached    */
    long long used_kb;      /* Calculated: total - available */
} MemInfo;

/*
 * DiskInfo - disk usage figures for a given mount point
 */
typedef struct {
    unsigned long long total_bytes;
    unsigned long long used_bytes;
    unsigned long long free_bytes;
    double usage_percent;
} DiskInfo;

/*
 * ProcessInfo - basic information about one running process
 *
 * This is populated by reading /proc/[pid]/stat and /proc/[pid]/status.
 */
typedef struct {
    pid_t  pid;                  /* Process ID (unique number)        */
    char   name[MAX_NAME_LEN];   /* Process name (comm)               */
    char   state;                /* Process state: R, S, D, Z, T, ... */
    pid_t  ppid;                 /* Parent PID                        */
    long   priority;             /* Scheduling priority               */
    long   nice;                 /* Nice value (-20 to 19)            */
    long   num_threads;          /* Number of threads                 */
    long   vm_rss_kb;            /* Resident Set Size in KB (RAM used)*/
} ProcessInfo;

/* ---------------------------------------------------------------
 * Function Declarations
 *
 * These are "prototypes" — they tell the compiler the name, return
 * type, and parameter types of every function defined in the .c files.
 * Without these, calling a function from a different .c file would
 * cause a compilation error.
 * --------------------------------------------------------------- */

/* --- cpu.c --- */
int  cpu_read_stats(CpuStats *out);
double cpu_calculate_usage(const CpuStats *prev, const CpuStats *curr);
void cpu_print_usage(void);

/* --- memory.c --- */
int  mem_read_info(MemInfo *out);
void mem_print_usage(void);

/* --- disk.c --- */
int  disk_read_info(const char *mount_point, DiskInfo *out);
void disk_print_usage(const char *mount_point);

/* --- process.c --- */
int  proc_is_pid_dir(const char *name);
int  proc_read_info(pid_t pid, ProcessInfo *out);
int  proc_list_all(ProcessInfo *list, int max_count);
void proc_print_list(const ProcessInfo *list, int count);
void proc_print_details(pid_t pid);
int  proc_terminate(pid_t pid);

/* --- main.c (UI helpers) --- */
void print_header(void);
void print_separator(void);

#endif /* MONITOR_H */
