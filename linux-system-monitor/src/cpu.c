/*
 * cpu.c - CPU usage monitoring via /proc/stat
 *
 * HOW CPU USAGE IS CALCULATED
 * ============================
 * The Linux kernel exposes cumulative CPU time counters in /proc/stat.
 * Because these counters only ever go up, we cannot get a percentage
 * from a single reading. Instead, we:
 *
 *   1. Read /proc/stat  →  record "snapshot 1"
 *   2. Sleep for a short time  (e.g., 500 ms)
 *   3. Read /proc/stat  →  record "snapshot 2"
 *   4. Compute the delta (difference) for each counter
 *   5. CPU% = (delta_busy / delta_total) * 100
 *
 * "busy" = user + nice + system + irq + softirq
 * "total" = busy + idle + iowait
 *
 * This is exactly how tools like `top` and `htop` work.
 *
 * Linux concept demonstrated:
 *   - Reading from the /proc virtual filesystem
 *   - Parsing structured text files
 *   - Using time-delta measurements for rates
 */

#include "../include/monitor.h"

/*
 * cpu_read_stats()
 * ----------------
 * Opens /proc/stat and reads the first line, which starts with "cpu"
 * and contains space-separated cumulative tick counts.
 *
 * Example line from /proc/stat:
 *   cpu  123456 789 456789 9876543 12345 0 678 0 0 0
 *         user  nice  sys   idle   iowait irq softirq ...
 *
 * Parameters:
 *   out  - pointer to a CpuStats struct that will be filled in
 *
 * Returns:
 *   0 on success, -1 on failure
 */
int cpu_read_stats(CpuStats *out)
{
    /* FILE* is the C standard library type for an open file stream. */
    FILE *fp = fopen("/proc/stat", "r");
    if (fp == NULL) {
        /* errno is set automatically by fopen() on failure. */
        perror("cpu_read_stats: fopen /proc/stat");
        return -1;
    }

    /*
     * sscanf parses a formatted string — here we skip the "cpu" label
     * and read 7 long long integers into our struct fields.
     */
    char line[LINE_BUF_SIZE];
    if (fgets(line, sizeof(line), fp) == NULL) {
        perror("cpu_read_stats: fgets");
        fclose(fp);
        return -1;
    }

    /*
     * Parse: "cpu  user nice system idle iowait irq softirq ..."
     * We use %*s to skip the "cpu" label token.
     */
    int parsed = sscanf(line,
        "%*s %lld %lld %lld %lld %lld %lld %lld",
        &out->user,
        &out->nice,
        &out->system,
        &out->idle,
        &out->iowait,
        &out->irq,
        &out->softirq);

    fclose(fp);

    if (parsed != 7) {
        fprintf(stderr, "cpu_read_stats: unexpected format in /proc/stat\n");
        return -1;
    }

    return 0;
}

/*
 * cpu_calculate_usage()
 * ----------------------
 * Given two CpuStats snapshots (taken before and after a sleep),
 * compute the CPU usage percentage over that interval.
 *
 * Parameters:
 *   prev  - snapshot taken BEFORE the sleep
 *   curr  - snapshot taken AFTER  the sleep
 *
 * Returns:
 *   CPU usage as a percentage (0.0 – 100.0), or -1.0 on error.
 */
double cpu_calculate_usage(const CpuStats *prev, const CpuStats *curr)
{
    /*
     * Compute the change in each counter between the two snapshots.
     * Delta values represent work done during our measurement window.
     */
    long long delta_user    = curr->user    - prev->user;
    long long delta_nice    = curr->nice    - prev->nice;
    long long delta_system  = curr->system  - prev->system;
    long long delta_idle    = curr->idle    - prev->idle;
    long long delta_iowait  = curr->iowait  - prev->iowait;
    long long delta_irq     = curr->irq     - prev->irq;
    long long delta_softirq = curr->softirq - prev->softirq;

    /* Total CPU time = busy time + idle time */
    long long delta_total = delta_user + delta_nice + delta_system
                          + delta_idle + delta_iowait
                          + delta_irq  + delta_softirq;

    /* Busy time = everything except idle and iowait */
    long long delta_busy  = delta_user + delta_nice + delta_system
                          + delta_irq  + delta_softirq;

    if (delta_total <= 0) {
        /* Protect against divide-by-zero if no time has passed */
        return 0.0;
    }

    /* Convert to percentage */
    return (double)delta_busy / (double)delta_total * 100.0;
}

/*
 * cpu_print_usage()
 * -----------------
 * High-level convenience function: reads CPU stats twice with a 500ms
 * delay and prints the resulting usage percentage to the terminal.
 */
void cpu_print_usage(void)
{
    CpuStats before, after;

    if (cpu_read_stats(&before) != 0) {
        printf("  CPU Usage      : [error reading /proc/stat]\n");
        return;
    }

    /*
     * usleep() pauses execution for the given number of MICROseconds.
     * 500,000 microseconds = 500 milliseconds = 0.5 seconds.
     * This gives the kernel enough time to accumulate meaningful delta.
     */
    usleep(500000);

    if (cpu_read_stats(&after) != 0) {
        printf("  CPU Usage      : [error reading /proc/stat]\n");
        return;
    }

    double usage = cpu_calculate_usage(&before, &after);
    printf("  CPU Usage      : %.1f%%\n", usage);
}
