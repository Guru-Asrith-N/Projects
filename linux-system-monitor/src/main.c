/*
 * main.c - Entry point and interactive menu for Linux System Monitor
 *
 * This file contains:
 *   - main()        : program entry, main menu loop
 *   - print_header(): decorative header for the terminal UI
 *   - show_dashboard(): quick summary of CPU, memory, disk
 *   - live_monitor(): continuously refreshing display mode
 *   - Each menu handler calls functions from cpu.c, memory.c,
 *     disk.c, and process.c via the declarations in monitor.h
 *
 * Design:
 *   - Simple do-while loop for the menu
 *   - fgets() + sscanf() for safe user input (avoids buffer overflows
 *     that scanf("%d") can cause)
 *   - ANSI escape codes for basic terminal formatting (clear screen)
 *
 * Linux concept demonstrated:
 *   - getpid() — every process knows its own PID
 *   - Signals in live monitor (Ctrl+C exits gracefully)
 */

#include "../include/monitor.h"

/* Width of decorative separator lines */
#define SEPARATOR_WIDTH 50

/* Mount point to monitor for disk usage */
#define DISK_MOUNT_POINT "/"

/* ---------------------------------------------------------------
 * Terminal UI Helpers
 * --------------------------------------------------------------- */

/*
 * print_separator()
 * Prints a horizontal line of dashes.
 */
void print_separator(void)
{
    printf("  ");
    for (int i = 0; i < SEPARATOR_WIDTH; i++) {
        putchar('-');
    }
    putchar('\n');
}

/*
 * print_header()
 * Prints the application title.
 */
void print_header(void)
{
    printf("\n");
    print_separator();
    printf("  LINUX SYSTEM MONITOR & PROCESS MANAGER\n");
    printf("  Running as PID: %d\n", getpid());
    print_separator();
}

/*
 * clear_screen()
 * Uses an ANSI escape sequence to clear the terminal.
 * "\033[2J"   clears the screen
 * "\033[H"    moves cursor to top-left
 */
static void clear_screen(void)
{
    printf("\033[2J\033[H");
    fflush(stdout);
}

/*
 * read_int_input()
 * ----------------
 * Safely reads an integer from the user using fgets + sscanf.
 *
 * WHY NOT scanf()?
 *   scanf("%d", &n) is dangerous because:
 *   1. It leaves a newline in the buffer, confusing the next read.
 *   2. Non-numeric input can corrupt the input stream.
 *   fgets() reads the whole line safely, then sscanf() parses it.
 *
 * Returns:
 *   The integer typed by the user, or -1 if input is invalid.
 */
static int read_int_input(void)
{
    char buf[64];
    if (fgets(buf, sizeof(buf), stdin) == NULL) {
        return -1;
    }
    int value = -1;
    sscanf(buf, "%d", &value);
    return value;
}

/* ---------------------------------------------------------------
 * Dashboard
 * --------------------------------------------------------------- */

/*
 * show_dashboard()
 * ----------------
 * Displays a quick summary of CPU, memory, and disk usage.
 * Called from the main menu and from the live monitor.
 *
 * NOTE: cpu_print_usage() internally sleeps for 500ms to take
 * two CPU snapshots. This means the dashboard takes ~0.5 s to display.
 */
static void show_dashboard(void)
{
    printf("\n  [ System Overview ]\n\n");
    cpu_print_usage();
    mem_print_usage();
    disk_print_usage(DISK_MOUNT_POINT);
    printf("\n");
}

/* ---------------------------------------------------------------
 * Live Monitor
 * --------------------------------------------------------------- */

/*
 * live_monitor()
 * --------------
 * Continuously refreshes the dashboard at MONITOR_REFRESH_MS intervals.
 * The user exits by pressing Ctrl+C.
 *
 * Implementation:
 *   - clear_screen() before each refresh
 *   - usleep() between refreshes
 *   - The loop runs until the user presses Ctrl+C, which sends
 *     SIGINT to our process. The default handler exits the program.
 *
 * Linux concept: Ctrl+C sends SIGINT (signal 2) to the foreground
 * process group. The default action for SIGINT is to terminate the
 * process. We could install a custom signal handler, but for simplicity
 * we rely on the default behavior here.
 */
static void live_monitor(void)
{
    printf("\n  Starting live monitor. Press Ctrl+C to return to menu.\n");
    printf("  Refreshing every %d ms...\n\n", MONITOR_REFRESH_MS);
    sleep(2); /* Brief pause so the user can read the instruction */

    int refresh_count = 0;

    while (1) {
        clear_screen();
        print_header();

        printf("  [ Live Monitor — Refresh #%d ]\n", ++refresh_count);
        printf("  Press Ctrl+C to exit live mode.\n");

        /* Display current system stats */
        printf("\n");
        cpu_print_usage();       /* Takes ~500ms internally */
        mem_print_usage();
        disk_print_usage(DISK_MOUNT_POINT);
        printf("\n");

        /*
         * List the top 15 processes by PID (simplest sort = no sort,
         * just whatever /proc/ returns first).
         */
        printf("  [ Recent Processes (first 15) ]\n");
        static ProcessInfo proc_list[MAX_PROCESSES];
        int count = proc_list_all(proc_list, MAX_PROCESSES);
        if (count > 0) {
            int show = (count < 15) ? count : 15;
            proc_print_list(proc_list, show);
        }

        /*
         * usleep() suspends execution for MONITOR_REFRESH_MS * 1000
         * microseconds. The CPU measurement in cpu_print_usage() already
         * adds ~500ms, so the effective refresh rate is approximately
         * MONITOR_REFRESH_MS + 500ms.
         */
        usleep((unsigned int)MONITOR_REFRESH_MS * 1000);
    }
}

/* ---------------------------------------------------------------
 * Menu Handlers
 * --------------------------------------------------------------- */

/* Handler: option 1 — System Information */
static void handle_system_info(void)
{
    print_header();
    show_dashboard();
    printf("  Press Enter to return to menu...");
    fflush(stdout);
    /* Wait for user to press Enter */
    char dummy[8];
    fgets(dummy, sizeof(dummy), stdin);
}

/* Handler: option 2 — List Running Processes */
static void handle_list_processes(void)
{
    print_header();
    printf("  [ Running Processes ]\n");
    printf("  Reading /proc/ directory...\n\n");

    /*
     * Declare a static array to hold process info.
     * We use static here so the array lives in BSS/data segment
     * instead of the stack — it's too large for the stack.
     */
    static ProcessInfo proc_list[MAX_PROCESSES];
    int count = proc_list_all(proc_list, MAX_PROCESSES);

    if (count < 0) {
        printf("  [Error] Failed to enumerate processes.\n");
    } else {
        proc_print_list(proc_list, count);
    }

    printf("\n  Press Enter to return to menu...");
    fflush(stdout);
    char dummy[8];
    fgets(dummy, sizeof(dummy), stdin);
}

/* Handler: option 3 — Process Details */
static void handle_process_details(void)
{
    print_header();
    printf("  [ Process Details ]\n\n");
    printf("  Enter PID to inspect (or 0 to cancel): ");
    fflush(stdout);

    int pid = read_int_input();
    if (pid <= 0) {
        printf("  Cancelled.\n");
    } else {
        proc_print_details((pid_t)pid);
    }

    printf("\n  Press Enter to return to menu...");
    fflush(stdout);
    char dummy[8];
    fgets(dummy, sizeof(dummy), stdin);
}

/* Handler: option 4 — Terminate Process */
static void handle_terminate_process(void)
{
    print_header();
    printf("  [ Terminate Process ]\n\n");
    printf("  WARNING: This will send SIGTERM to the chosen process.\n");
    printf("  Only terminate processes you own (or run as root).\n\n");
    printf("  Enter PID to terminate (or 0 to cancel): ");
    fflush(stdout);

    int pid = read_int_input();
    if (pid <= 0) {
        printf("  Cancelled.\n");
    } else {
        /*
         * Before terminating, show the user what process they are about
         * to kill so they can verify it's the right one.
         */
        ProcessInfo info;
        if (proc_read_info((pid_t)pid, &info) == 0) {
            printf("\n  You are about to terminate:\n");
            printf("    PID  : %d\n", info.pid);
            printf("    Name : %s\n", info.name);
            printf("\n  Are you sure? Type 'yes' to confirm: ");
            fflush(stdout);

            char confirm[16];
            fgets(confirm, sizeof(confirm), stdin);

            /* Only proceed if the user explicitly types "yes" */
            if (strncmp(confirm, "yes", 3) == 0) {
                proc_terminate((pid_t)pid);
            } else {
                printf("  Cancelled.\n");
            }
        } else {
            printf("  [Error] PID %d not found or not accessible.\n", pid);
        }
    }

    printf("\n  Press Enter to return to menu...");
    fflush(stdout);
    char dummy[8];
    fgets(dummy, sizeof(dummy), stdin);
}

/* Handler: option 5 — Live Monitor */
static void handle_live_monitor(void)
{
    live_monitor();
    /* Control returns here only if live_monitor() is interrupted by Ctrl+C.
     * In our current implementation, Ctrl+C exits the whole program.
     * A more advanced implementation would install a SIGINT handler. */
}

/* ---------------------------------------------------------------
 * main()
 * ---------------------------------------------------------------
 * Program entry point. Displays the main menu in a loop until the
 * user selects "Exit".
 */
int main(void)
{
    int choice = 0;

    do {
        clear_screen();
        print_header();
        show_dashboard();

        /* Print the menu */
        print_separator();
        printf("  MENU\n");
        print_separator();
        printf("  1. System Information\n");
        printf("  2. Running Processes\n");
        printf("  3. Process Details\n");
        printf("  4. Terminate Process\n");
        printf("  5. Live Monitor\n");
        printf("  6. Exit\n");
        print_separator();
        printf("  Enter choice [1-6]: ");
        fflush(stdout);

        choice = read_int_input();

        switch (choice) {
            case 1:
                handle_system_info();
                break;
            case 2:
                handle_list_processes();
                break;
            case 3:
                handle_process_details();
                break;
            case 4:
                handle_terminate_process();
                break;
            case 5:
                handle_live_monitor();
                break;
            case 6:
                clear_screen();
                printf("\n  Goodbye!\n\n");
                break;
            default:
                printf("\n  [!] Invalid choice. Please enter 1-6.\n");
                sleep(1);
                break;
        }

    } while (choice != 6);

    return 0;
}
