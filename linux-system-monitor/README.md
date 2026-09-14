# Linux System Monitor & Process Manager

A beginner-friendly command-line application written in **C** that demonstrates core Linux system programming concepts — including `/proc` filesystem access, system calls, process management, and signal handling.

Built as a learning project for final-year computer science students who want to understand how Linux exposes operating system internals.

---

## Features

| # | Feature | How It Works |
|---|---------|--------------|
| 1 | **CPU Usage** | Reads `/proc/stat` twice, computes the delta |
| 2 | **Memory Usage** | Parses `/proc/meminfo` |
| 3 | **Disk Usage** | Calls `statvfs()` system call |
| 4 | **List Processes** | Scans `/proc/` directory for numeric subdirs |
| 5 | **Process Details** | Reads `/proc/[pid]/stat` and `/proc/[pid]/status` |
| 6 | **Terminate Process** | Uses `kill(pid, SIGTERM)` system call |
| 7 | **Live Monitor** | Continuously refreshes with `usleep()` |

---

## Technologies

- **Language**: C (C11 standard)
- **APIs**: Linux/POSIX
- **Virtual Filesystem**: `/proc`
- **Key System Calls**: `statvfs()`, `kill()`, `opendir()`, `readdir()`
- **Build Tool**: GNU Make
- **Compiler**: GCC

**No external libraries** — only the C standard library and POSIX headers.

---

## Project Structure

```
linux-system-monitor/
├── src/
│   ├── main.c      ← Menu, UI, live monitor
│   ├── cpu.c       ← CPU usage from /proc/stat
│   ├── memory.c    ← Memory usage from /proc/meminfo
│   ├── disk.c      ← Disk usage via statvfs()
│   └── process.c   ← Process listing, details, termination
├── include/
│   └── monitor.h   ← Shared types, constants, function declarations
├── scripts/
│   └── build.sh    ← Alternative build script (no make required)
├── Makefile
├── README.md
├── INTERVIEW_GUIDE.md
├── 3_DAY_LEARNING_PLAN.md
└── .gitignore
```

---

## How `/proc` Works

`/proc` is a **virtual filesystem** — it exists only in memory and is created by the Linux kernel at boot time. When you read a file like `/proc/meminfo`, no disk is accessed. The kernel generates the content on the fly in response to the `read()` system call.

This design (called the **Unix philosophy**: "everything is a file") lets any program — written in any language — query the operating system simply by opening and reading a text file. No special API is needed.

Key `/proc` paths used in this project:

| Path | Contents |
|------|----------|
| `/proc/stat` | Cumulative CPU time counters |
| `/proc/meminfo` | Memory and swap statistics |
| `/proc/[pid]/stat` | Process status (one line, 52 fields) |
| `/proc/[pid]/status` | Human-readable process details |
| `/proc/[pid]/cmdline` | Command line used to start the process |

---

## How CPU Usage Is Calculated

CPU usage cannot be measured from a single reading because `/proc/stat` only provides **cumulative** (ever-increasing) counters — not an instantaneous rate.

**Algorithm:**

1. Read `/proc/stat` → store counters as **snapshot A**
2. Sleep for 500 milliseconds
3. Read `/proc/stat` again → store counters as **snapshot B**
4. Compute the delta (difference) for each counter
5. `CPU% = (delta_busy / delta_total) × 100`

Where:
- `busy = user + nice + system + irq + softirq`
- `total = busy + idle + iowait`

This is exactly how tools like `top`, `htop`, and `vmstat` work.

```c
double cpu_calculate_usage(const CpuStats *prev, const CpuStats *curr)
{
    long long delta_busy  = (curr->user - prev->user)
                          + (curr->system - prev->system)
                          + /* ... other non-idle fields ... */ ;
    long long delta_total = delta_busy + (curr->idle - prev->idle)
                          + (curr->iowait - prev->iowait);
    return (double)delta_busy / (double)delta_total * 100.0;
}
```

---

## How Memory Usage Is Calculated

We read `/proc/meminfo` and extract:

| Field | Meaning |
|-------|---------|
| `MemTotal` | Total physical RAM |
| `MemAvailable` | RAM available for new processes (free + reclaimable cache) |

```
Used Memory = MemTotal - MemAvailable
Usage %     = (Used / Total) × 100
```

**Important**: We use `MemAvailable` (not `MemFree`) because Linux aggressively uses "free" RAM as disk cache to improve performance. `MemAvailable` accounts for this reclaimable cache and gives a more meaningful "how much RAM can I use?" answer.

---

## How Processes Are Discovered

```
Algorithm (O(N) where N = entries in /proc/):

1. Open /proc as a directory: opendir("/proc")
2. Loop: entry = readdir(proc_dir)
3. If entry->d_name is all digits → it's a PID directory
4. Read /proc/[pid]/stat → extract process info
5. Store in ProcessInfo array
6. closedir(proc_dir)
```

Only numeric directory names correspond to processes. Entries like `net`, `sys`, `bus`, and `self` are filtered out by checking if every character in the name is a digit.

---

## How Process Termination Works

```c
kill(pid, SIGTERM);
```

`kill()` is a POSIX system call that sends a **signal** to a process. Despite its name, it can send *any* signal — not just termination signals.

**SIGTERM (signal 15)** is the standard "please stop" request:
- The target process *can* catch it and perform cleanup (close files, flush buffers).
- It is the polite equivalent of "please exit when you're ready."

**Why it can fail:**
| Error | Reason |
|-------|--------|
| `EPERM` | You don't own the process; you need `sudo` |
| `ESRCH` | The PID doesn't exist (process already exited) |
| Signal ignored | The process has a custom SIGTERM handler that ignores it |

This program always requires the user to manually type the PID and confirm with "yes" before sending the signal.

---

## Compilation

### Option 1 — Using Make (recommended)

```bash
make
```

The binary will be created at `bin/system_monitor`.

### Option 2 — Using the build script

```bash
chmod +x scripts/build.sh
./scripts/build.sh
```

### Option 3 — Manual compilation

```bash
mkdir -p bin obj
gcc -Wall -Wextra -g -std=c11 -I include -c src/cpu.c     -o obj/cpu.o
gcc -Wall -Wextra -g -std=c11 -I include -c src/memory.c  -o obj/memory.o
gcc -Wall -Wextra -g -std=c11 -I include -c src/disk.c    -o obj/disk.o
gcc -Wall -Wextra -g -std=c11 -I include -c src/process.c -o obj/process.o
gcc -Wall -Wextra -g -std=c11 -I include -c src/main.c    -o obj/main.o
gcc -Wall -Wextra -g -std=c11 -I include -o bin/system_monitor \
    obj/main.o obj/cpu.o obj/memory.o obj/disk.o obj/process.o
```

### Requirements

- Linux (any distribution)
- GCC (`sudo apt install gcc` on Ubuntu/Debian)
- Make (`sudo apt install make` on Ubuntu/Debian)

---

## Running

```bash
./bin/system_monitor
```

Or with make:

```bash
make run
```

---

## Example Output

```
  --------------------------------------------------
  LINUX SYSTEM MONITOR & PROCESS MANAGER
  Running as PID: 12345
  --------------------------------------------------

  [ System Overview ]

  CPU Usage      : 12.4%
  Memory Usage   : 47.2% (7.56 GB used / 16.00 GB total)
  Disk Usage (/) : 38.1% (114.3 GB used / 300.0 GB total, 185.7 GB free)

  --------------------------------------------------
  MENU
  --------------------------------------------------
  1. System Information
  2. Running Processes
  3. Process Details
  4. Terminate Process
  5. Live Monitor
  6. Exit
  --------------------------------------------------
  Enter choice [1-6]:
```

---

## Limitations

- **Root required** to terminate other users' processes
- **CPU measurement** takes ~500ms due to the two-snapshot approach
- **Process list** is a point-in-time snapshot; processes can appear/disappear
- **No sorting** of the process list (processes appear in arbitrary /proc/ order)
- **Single disk** monitored (`/` only)
- **No network** monitoring
- **No historical graphs** — only current readings
- **Not portable** to macOS or Windows (uses Linux-specific `/proc`)

---

## Possible Future Improvements

1. **Sort process list** by CPU or memory usage
2. **Filter processes** by name or user
3. **Color output** using ANSI color codes
4. **Network statistics** from `/proc/net/dev`
5. **Per-CPU core** breakdown (read `cpu0`, `cpu1`, ... lines in `/proc/stat`)
6. **SIGKILL option** for stubborn processes
7. **Save output** to a log file
8. **ncurses UI** for a proper terminal dashboard
9. **Multiple disk** mount points
10. **Swap usage** from `/proc/meminfo`

---

## License

MIT — free to use, modify, and distribute.
