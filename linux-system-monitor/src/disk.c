/*
 * disk.c - Disk usage monitoring via statvfs()
 *
 * HOW DISK USAGE IS READ
 * =======================
 * Unlike CPU and memory, disk usage is NOT in /proc. Instead, we use
 * the POSIX statvfs() system call, which asks the kernel for filesystem
 * statistics about a given mount point (directory).
 *
 * statvfs() fills a struct statvfs with:
 *   f_blocks  - total number of blocks in the filesystem
 *   f_bfree   - number of free blocks
 *   f_bavail  - free blocks available to non-root users
 *   f_bsize   - fundamental block size (bytes per block)
 *   f_frsize  - fragment size (use this for capacity calculations)
 *
 * Total bytes  = f_blocks * f_frsize
 * Free  bytes  = f_bavail * f_frsize   (non-root usable)
 * Used  bytes  = total - free
 *
 * We monitor "/" (the root filesystem) by default. On a typical system
 * this covers the main disk partition.
 *
 * Linux concepts demonstrated:
 *   - System call: statvfs()
 *   - Filesystem concepts: blocks, mount points
 *   - Unsigned integer arithmetic for byte counts
 */

#include "../include/monitor.h"

/*
 * disk_read_info()
 * ----------------
 * Calls statvfs() on the given mount point and fills a DiskInfo struct.
 *
 * Parameters:
 *   mount_point  - path to the directory to examine (e.g., "/")
 *   out          - pointer to DiskInfo struct to fill
 *
 * Returns:
 *   0 on success, -1 on failure
 */
int disk_read_info(const char *mount_point, DiskInfo *out)
{
    struct statvfs stat;

    /*
     * statvfs() is a POSIX system call that fills 'stat' with
     * filesystem statistics for the filesystem containing 'mount_point'.
     * It returns 0 on success, -1 on error (sets errno).
     */
    if (statvfs(mount_point, &stat) != 0) {
        perror("disk_read_info: statvfs");
        return -1;
    }

    /*
     * f_frsize is the "fragment size" — the real block granularity.
     * Multiply by block counts to get byte totals.
     * We cast to unsigned long long to avoid overflow for large disks.
     */
    out->total_bytes = (unsigned long long)stat.f_blocks * stat.f_frsize;
    out->free_bytes  = (unsigned long long)stat.f_bavail * stat.f_frsize;
    out->used_bytes  = out->total_bytes - out->free_bytes;

    /* Guard against divide-by-zero for zero-size filesystems */
    if (out->total_bytes > 0) {
        out->usage_percent = (double)out->used_bytes
                           / (double)out->total_bytes * 100.0;
    } else {
        out->usage_percent = 0.0;
    }

    return 0;
}

/*
 * disk_print_usage()
 * ------------------
 * Reads and prints disk usage for the given mount point in a
 * human-readable format (GB).
 *
 * Parameters:
 *   mount_point  - path to examine (typically "/")
 */
void disk_print_usage(const char *mount_point)
{
    DiskInfo disk;

    if (disk_read_info(mount_point, &disk) != 0) {
        printf("  Disk Usage     : [error calling statvfs]\n");
        return;
    }

    /* Convert bytes to gigabytes */
    double total_gb = (double)disk.total_bytes / (1024.0 * 1024.0 * 1024.0);
    double used_gb  = (double)disk.used_bytes  / (1024.0 * 1024.0 * 1024.0);
    double free_gb  = (double)disk.free_bytes  / (1024.0 * 1024.0 * 1024.0);

    printf("  Disk Usage (%s) : %.1f%% (%.1f GB used / %.1f GB total, %.1f GB free)\n",
           mount_point, disk.usage_percent, used_gb, total_gb, free_gb);
}
