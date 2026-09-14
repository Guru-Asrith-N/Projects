/*
 * memory.c - Memory usage monitoring via /proc/meminfo
 *
 * HOW LINUX MEMORY WORKS (simplified)
 * =====================================
 * Linux manages physical RAM in pages (usually 4 KB each).
 * The kernel tracks how much RAM is:
 *   - Free           : completely unused
 *   - Used by apps   : actively mapped into process address spaces
 *   - Cached/Buffered: used as disk cache but reclaimable
 *   - Available      : free + reclaimable (what apps can actually use)
 *
 * We read these figures from /proc/meminfo, which the kernel updates
 * in real time. All values are in kilobytes (KB).
 *
 * KEY INSIGHT:
 *   "Used" memory = MemTotal - MemAvailable
 *   This is the most accurate "how much RAM am I actually using?" metric.
 *   Older tools used (MemTotal - MemFree) which gave misleading results
 *   because they didn't account for reclaimable cache.
 *
 * Linux concepts demonstrated:
 *   - /proc/meminfo parsing
 *   - sscanf for structured text parsing
 *   - KB to GB/MB conversion
 */

#include "../include/monitor.h"

/*
 * mem_read_info()
 * ---------------
 * Reads /proc/meminfo line by line and extracts the fields we need.
 *
 * Example lines from /proc/meminfo:
 *   MemTotal:       16202140 kB
 *   MemFree:         1234567 kB
 *   MemAvailable:    5678901 kB
 *   Buffers:          345678 kB
 *   Cached:          2345678 kB
 *
 * Parameters:
 *   out  - pointer to a MemInfo struct that will be filled in
 *
 * Returns:
 *   0 on success, -1 on failure
 */
int mem_read_info(MemInfo *out)
{
    FILE *fp = fopen("/proc/meminfo", "r");
    if (fp == NULL) {
        perror("mem_read_info: fopen /proc/meminfo");
        return -1;
    }

    /* Zero out the struct so any un-filled field defaults to 0 */
    memset(out, 0, sizeof(MemInfo));

    char line[LINE_BUF_SIZE];
    char key[64];
    long long value;

    /*
     * Read /proc/meminfo line by line.
     * Each line is: "KeyName:   12345 kB"
     * We parse the key name and value, then store the ones we want.
     */
    while (fgets(line, sizeof(line), fp) != NULL) {
        /* Parse "Key: value kB" — we ignore the "kB" unit string */
        if (sscanf(line, "%63s %lld", key, &value) != 2) {
            continue; /* Skip lines that don't match our pattern */
        }

        if (strcmp(key, "MemTotal:") == 0) {
            out->total_kb = value;
        } else if (strcmp(key, "MemFree:") == 0) {
            out->free_kb = value;
        } else if (strcmp(key, "MemAvailable:") == 0) {
            out->available_kb = value;
        } else if (strcmp(key, "Buffers:") == 0) {
            out->buffers_kb = value;
        } else if (strcmp(key, "Cached:") == 0) {
            out->cached_kb = value;
        }
    }

    fclose(fp);

    /* Sanity check: MemTotal must be non-zero */
    if (out->total_kb == 0) {
        fprintf(stderr, "mem_read_info: could not parse MemTotal\n");
        return -1;
    }

    /*
     * Calculate used memory.
     * MemAvailable is a kernel-provided estimate of how much RAM
     * is available for new processes (accounts for reclaimable cache).
     * used = total - available  gives the most meaningful "used" number.
     */
    out->used_kb = out->total_kb - out->available_kb;

    return 0;
}

/*
 * mem_print_usage()
 * -----------------
 * Reads and prints a formatted memory usage summary.
 * Converts KB figures to GB for readability.
 */
void mem_print_usage(void)
{
    MemInfo mem;

    if (mem_read_info(&mem) != 0) {
        printf("  Memory Usage   : [error reading /proc/meminfo]\n");
        return;
    }

    /* Convert kilobytes to gigabytes for display */
    double total_gb = (double)mem.total_kb / (1024.0 * 1024.0);
    double used_gb  = (double)mem.used_kb  / (1024.0 * 1024.0);
    double pct      = (double)mem.used_kb  / (double)mem.total_kb * 100.0;

    printf("  Memory Usage   : %.1f%% (%.2f GB used / %.2f GB total)\n",
           pct, used_gb, total_gb);
}
