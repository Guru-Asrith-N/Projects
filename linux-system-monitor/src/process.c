/*
 * process.c - Process enumeration, inspection, and termination
 *
 * HOW LINUX PROCESSES WORK
 * =========================
 * Every running program on Linux is a "process". Each process is
 * assigned a unique integer called a PID (Process ID).
 *
 * The kernel exposes information about every running process through
 * the /proc virtual filesystem. For each process with PID N, there is
 * a directory /proc/N/ containing many files, for example:
 *
 *   /proc/N/stat    - One line: PID, name, state, PPID, priority, etc.
 *   /proc/N/status  - Human-readable version of process info
 *   /proc/N/cmdline - Full command line used to start the process
 *   /proc/N/maps    - Memory mappings
 *   /proc/N/fd/     - Directory listing open file descriptors
 *
 * To list ALL processes, we open /proc/ as a directory and look for
 * sub-directories whose names are purely numeric (e.g., "1", "42", "3871").
 * Only PID directories have numeric names.
 *
 * HOW PROCESS TERMINATION WORKS
 * ==============================
 * We use the kill() system call to send a signal to a process.
 * Signals are integer messages the kernel delivers to processes.
 *
 * kill(pid, SIGTERM)  — asks the process to terminate gracefully
 * kill(pid, SIGKILL)  — forces immediate termination (cannot be caught)
 *
 * This program only uses SIGTERM (the polite request).
 * The process itself can catch SIGTERM and do cleanup before exiting.
 *
 * Linux concepts demonstrated:
 *   - opendir() / readdir() / closedir()  for directory traversal
 *   - /proc/[pid]/stat parsing
 *   - kill() system call
 *   - SIGTERM signal
 *   - errno and permission error handling
 */

#include "../include/monitor.h"

/* ---------------------------------------------------------------
 * Internal Helper: Read a Single Line from a /proc File
 * --------------------------------------------------------------- */

/*
 * read_first_line()
 * -----------------
 * Opens a file, reads its first line into 'buf', and closes the file.
 * Used internally to read small /proc files.
 *
 * Returns: 0 on success, -1 on failure.
 */
static int read_first_line(const char *path, char *buf, int buf_size)
{
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    if (fgets(buf, buf_size, fp) == NULL) {
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}

/* ---------------------------------------------------------------
 * proc_is_pid_dir()
 * ---------------------------------------------------------------
 * Returns 1 if the given directory name is a valid PID (all digits),
 * or 0 if it is not (e.g., "net", "sys", "self", etc.)
 *
 * We use this to filter the entries in /proc/ so we only look at
 * process directories, not the many other /proc subdirectories.
 *
 * Example:
 *   proc_is_pid_dir("1234")  → 1   (it's a PID)
 *   proc_is_pid_dir("net")   → 0   (it's not)
 *   proc_is_pid_dir("self")  → 0   (it's not)
 */
int proc_is_pid_dir(const char *name)
{
    if (name == NULL || *name == '\0') {
        return 0;
    }
    /* Every character must be a digit */
    for (int i = 0; name[i] != '\0'; i++) {
        if (!isdigit((unsigned char)name[i])) {
            return 0;
        }
    }
    return 1;
}

/* ---------------------------------------------------------------
 * proc_read_info()
 * ---------------------------------------------------------------
 * Reads /proc/[pid]/stat to fill a ProcessInfo struct.
 *
 * The format of /proc/[pid]/stat is a single line with space-separated
 * fields. The most important fields (1-indexed) are:
 *
 *   Field 1  : pid         — Process ID
 *   Field 2  : comm        — Process name in parentheses, e.g. "(bash)"
 *   Field 3  : state       — Single character: R, S, D, Z, T, etc.
 *   Field 4  : ppid        — Parent process ID
 *   Field 18 : priority    — Scheduling priority
 *   Field 19 : nice        — Nice value
 *   Field 20 : num_threads — Number of threads in the process
 *
 * VmRSS (physical memory used) comes from /proc/[pid]/status instead,
 * because it is easier to parse there.
 *
 * Parameters:
 *   pid  - the process ID to inspect
 *   out  - pointer to ProcessInfo struct to fill
 *
 * Returns:
 *   0 on success, -1 if the process cannot be read (may have exited)
 */
int proc_read_info(pid_t pid, ProcessInfo *out)
{
    char path[PROC_PATH_LEN];
    char line[LINE_BUF_SIZE];

    /* Build the path: "/proc/<pid>/stat" */
    snprintf(path, sizeof(path), "/proc/%d/stat", pid);

    if (read_first_line(path, line, sizeof(line)) != 0) {
        /* The process may have exited between enumeration and reading */
        return -1;
    }

    /* Zero out the output struct */
    memset(out, 0, sizeof(ProcessInfo));
    out->pid = pid;

    /*
     * Parse /proc/[pid]/stat
     *
     * The process name (comm) can contain spaces and is enclosed in
     * parentheses: "(my process)". We need to handle this carefully.
     *
     * Strategy:
     *   1. Find the first '(' and last ')' in the line.
     *   2. Extract the name between them.
     *   3. Parse the remaining fields after the closing ')'.
     */
    char *name_start = strchr(line, '(');
    char *name_end   = strrchr(line, ')');

    if (name_start == NULL || name_end == NULL || name_end <= name_start) {
        return -1;
    }

    /* Copy name (skip the '(' and stop before ')') */
    int name_len = (int)(name_end - name_start - 1);
    if (name_len >= MAX_NAME_LEN) {
        name_len = MAX_NAME_LEN - 1;
    }
    strncpy(out->name, name_start + 1, name_len);
    out->name[name_len] = '\0';

    /*
     * Parse the fields that appear AFTER the closing ')'.
     * name_end points to ')'; name_end+2 skips ") " to the state field.
     *
     * Remaining fields (relative to name_end+2):
     *   %c  = state
     *   %d  = ppid
     *   skip 13 fields (pgrp, session, tty_nr, tpgid, flags, minflt,
     *                    cminflt, majflt, cmajflt, utime, stime, cutime,
     *                    cstime)
     *   %ld = priority
     *   %ld = nice
     *   %ld = num_threads
     */
    char state;
    int  ppid;
    long priority, nice_val, num_threads;

    int matched = sscanf(name_end + 2,
        "%c "          /* state      */
        "%d "          /* ppid       */
        "%*d %*d %*d %*d %*u "          /* pgrp session tty_nr tpgid flags */
        "%*u %*u %*u %*u "              /* minflt cminflt majflt cmajflt   */
        "%*u %*u %*d %*d "              /* utime stime cutime cstime        */
        "%ld %ld %ld",                  /* priority nice num_threads        */
        &state, &ppid, &priority, &nice_val, &num_threads);

    if (matched < 2) {
        /* At minimum we need state and ppid */
        return -1;
    }

    out->state       = state;
    out->ppid        = (pid_t)ppid;
    out->priority    = (matched >= 3) ? priority    : 0;
    out->nice        = (matched >= 4) ? nice_val    : 0;
    out->num_threads = (matched >= 5) ? num_threads : 1;

    /*
     * Read VmRSS from /proc/[pid]/status
     * VmRSS = Resident Set Size = how much physical RAM the process uses
     *
     * /proc/[pid]/status has lines like:
     *   VmRSS:    12345 kB
     */
    snprintf(path, sizeof(path), "/proc/%d/status", pid);
    FILE *fp = fopen(path, "r");
    if (fp != NULL) {
        char key[64];
        long long value;
        char status_line[LINE_BUF_SIZE];
        while (fgets(status_line, sizeof(status_line), fp) != NULL) {
            if (sscanf(status_line, "%63s %lld", key, &value) == 2) {
                if (strcmp(key, "VmRSS:") == 0) {
                    out->vm_rss_kb = (long)value;
                    break;
                }
            }
        }
        fclose(fp);
    }

    return 0;
}

/* ---------------------------------------------------------------
 * proc_list_all()
 * ---------------------------------------------------------------
 * Enumerates all running processes by scanning the /proc directory.
 *
 * Algorithm:
 *   1. Open /proc as a directory using opendir()
 *   2. Iterate through all entries using readdir()
 *   3. For each entry whose name is numeric, call proc_read_info()
 *   4. Store results in the caller-provided array
 *
 * Parameters:
 *   list       - caller-allocated array of ProcessInfo structs
 *   max_count  - maximum number of entries in list[]
 *
 * Returns:
 *   Number of processes found (>=0), or -1 on fatal error.
 *
 * Time complexity: O(N) where N = number of entries in /proc/
 * Space complexity: O(P) where P = number of processes found
 */
int proc_list_all(ProcessInfo *list, int max_count)
{
    /*
     * opendir() opens a directory stream — similar to fopen() but
     * for directories. Returns a DIR* pointer, or NULL on failure.
     */
    DIR *proc_dir = opendir("/proc");
    if (proc_dir == NULL) {
        perror("proc_list_all: opendir /proc");
        return -1;
    }

    int count = 0;
    struct dirent *entry;

    /*
     * readdir() returns one directory entry at a time.
     * Each entry has a d_name field (the filename as a string).
     * Returns NULL when there are no more entries.
     *
     * Linux concept: struct dirent is defined in <dirent.h> and
     * represents a single directory entry.
     */
    while ((entry = readdir(proc_dir)) != NULL && count < max_count) {
        /*
         * Filter: only process directories have numeric names.
         * Skip ".", "..", "net", "sys", "bus", etc.
         */
        if (!proc_is_pid_dir(entry->d_name)) {
            continue;
        }

        /* Convert the directory name string to an integer PID */
        pid_t pid = (pid_t)atoi(entry->d_name);

        /*
         * Try to read process info. If it fails (e.g., the process
         * exited between readdir() and proc_read_info()), skip it.
         * This is normal and not an error.
         */
        if (proc_read_info(pid, &list[count]) == 0) {
            count++;
        }
    }

    /* Always close the directory stream to free resources */
    closedir(proc_dir);

    return count;
}

/* ---------------------------------------------------------------
 * proc_print_list()
 * ---------------------------------------------------------------
 * Prints a formatted table of all processes.
 */
void proc_print_list(const ProcessInfo *list, int count)
{
    printf("\n");
    /* Header row */
    printf("  %-8s %-6s %-20s %-8s %-8s\n",
           "PID", "State", "Name", "RSS(KB)", "Threads");
    printf("  %-8s %-6s %-20s %-8s %-8s\n",
           "--------", "------", "--------------------", "--------", "-------");

    for (int i = 0; i < count; i++) {
        printf("  %-8d %-6c %-20s %-8ld %-8ld\n",
               list[i].pid,
               list[i].state,
               list[i].name,
               list[i].vm_rss_kb,
               list[i].num_threads);
    }

    printf("\n  Total processes: %d\n", count);
}

/* ---------------------------------------------------------------
 * proc_print_details()
 * ---------------------------------------------------------------
 * Reads and prints detailed information about one specific process.
 *
 * Also reads /proc/[pid]/cmdline — the full command used to start
 * the process. In cmdline, arguments are separated by null bytes (\0),
 * so we replace them with spaces for display.
 */
void proc_print_details(pid_t pid)
{
    ProcessInfo info;

    if (proc_read_info(pid, &info) != 0) {
        printf("\n  [!] Could not read info for PID %d.\n", pid);
        printf("      The process may not exist or you may lack permission.\n");
        return;
    }

    /*
     * Process state codes (from the Linux kernel):
     *   R - Running (on CPU or in run queue)
     *   S - Sleeping (interruptible; waiting for event)
     *   D - Disk sleep (uninterruptible; waiting for I/O)
     *   Z - Zombie (finished but parent hasn't collected exit status)
     *   T - Stopped (by a signal or debugger)
     *   t - Tracing stop
     *   X - Dead
     */
    const char *state_desc;
    switch (info.state) {
        case 'R': state_desc = "Running";           break;
        case 'S': state_desc = "Sleeping";          break;
        case 'D': state_desc = "Disk Sleep (I/O)";  break;
        case 'Z': state_desc = "Zombie";             break;
        case 'T': state_desc = "Stopped";            break;
        default:  state_desc = "Unknown";            break;
    }

    printf("\n");
    printf("  ----------------------------------------\n");
    printf("  Process Details — PID %d\n", pid);
    printf("  ----------------------------------------\n");
    printf("  Name          : %s\n",   info.name);
    printf("  PID           : %d\n",   info.pid);
    printf("  Parent PID    : %d\n",   info.ppid);
    printf("  State         : %c (%s)\n", info.state, state_desc);
    printf("  Priority      : %ld\n",  info.priority);
    printf("  Nice Value    : %ld\n",  info.nice);
    printf("  Threads       : %ld\n",  info.num_threads);
    printf("  Memory (RSS)  : %ld KB\n", info.vm_rss_kb);

    /* Read and display the full command line */
    char cmd_path[PROC_PATH_LEN];
    snprintf(cmd_path, sizeof(cmd_path), "/proc/%d/cmdline", pid);
    FILE *fp = fopen(cmd_path, "r");
    if (fp != NULL) {
        char cmdline[512];
        int  len = (int)fread(cmdline, 1, sizeof(cmdline) - 1, fp);
        fclose(fp);
        if (len > 0) {
            cmdline[len] = '\0';
            /*
             * In /proc/[pid]/cmdline, arguments are separated by \0 bytes.
             * Replace each \0 with a space for readable display.
             */
            for (int i = 0; i < len - 1; i++) {
                if (cmdline[i] == '\0') {
                    cmdline[i] = ' ';
                }
            }
            printf("  Command       : %s\n", cmdline);
        }
    }

    printf("  ----------------------------------------\n");
}

/* ---------------------------------------------------------------
 * proc_terminate()
 * ---------------------------------------------------------------
 * Sends SIGTERM to the process with the given PID.
 *
 * SIGTERM (signal number 15) is the standard "please exit" signal.
 * Unlike SIGKILL (9), SIGTERM CAN be caught by the process, allowing
 * it to clean up resources (close files, release locks) before exiting.
 *
 * WHY TERMINATION CAN FAIL:
 *   1. EPERM  - We don't have permission. We can only kill our own
 *               processes (or all processes if we are root).
 *   2. ESRCH  - No process with that PID exists.
 *   3. The process is in state D (uninterruptible sleep) — SIGTERM
 *      will be queued but may not be processed.
 *   4. The process explicitly ignores SIGTERM in its signal handler.
 *
 * Parameters:
 *   pid  - PID of the process to terminate
 *
 * Returns:
 *   0 on success (signal sent), -1 on failure
 */
int proc_terminate(pid_t pid)
{
    /*
     * Safety check: never allow terminating PID 1 (init/systemd)
     * or PID 0 (the kernel itself), as this would crash the system.
     */
    if (pid <= 1) {
        printf("\n  [!] Refusing to terminate PID %d — system process.\n", pid);
        return -1;
    }

    printf("\n  Sending SIGTERM to PID %d ...\n", pid);

    /*
     * kill() is the system call for sending signals to processes.
     * Prototype: int kill(pid_t pid, int sig);
     *   pid  - target process ID
     *   sig  - signal number (SIGTERM = 15)
     * Returns 0 on success, -1 on error (sets errno).
     *
     * Despite its name, kill() can send ANY signal — it is not just
     * for termination.
     */
    if (kill(pid, SIGTERM) == 0) {
        printf("  [OK] SIGTERM sent to PID %d successfully.\n", pid);
        printf("       The process should exit shortly.\n");
        return 0;
    }

    /*
     * kill() failed. Check errno to give the user a clear message.
     * errno is a global variable set by system calls to indicate errors.
     */
    switch (errno) {
        case EPERM:
            printf("  [Error] Permission denied.\n");
            printf("          You can only terminate processes you own.\n");
            printf("          Try running this program with 'sudo' for\n");
            printf("          other users' processes.\n");
            break;
        case ESRCH:
            printf("  [Error] No process found with PID %d.\n", pid);
            printf("          It may have already exited.\n");
            break;
        default:
            printf("  [Error] kill() failed: %s (errno=%d)\n",
                   strerror(errno), errno);
            break;
    }

    return -1;
}
