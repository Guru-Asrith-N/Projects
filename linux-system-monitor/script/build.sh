#!/bin/bash
#
# build.sh - Simple build script for Linux System Monitor
#
# This script does the same thing as running `make` but can be run
# directly without having the `make` tool installed.
#
# Usage:
#   chmod +x scripts/build.sh
#   ./scripts/build.sh

set -e   # Exit immediately if any command returns a non-zero exit code

# Move to the project root (one directory above this script)
cd "$(dirname "$0")/.."

echo "============================================"
echo " Linux System Monitor — Build Script"
echo "============================================"

# Create output directories
mkdir -p bin obj

CC="gcc"
CFLAGS="-Wall -Wextra -g -std=c11 -I include"

echo "[1/5] Compiling cpu.c ..."
$CC $CFLAGS -c src/cpu.c -o obj/cpu.o

echo "[2/5] Compiling memory.c ..."
$CC $CFLAGS -c src/memory.c -o obj/memory.o

echo "[3/5] Compiling disk.c ..."
$CC $CFLAGS -c src/disk.c -o obj/disk.o

echo "[4/5] Compiling process.c ..."
$CC $CFLAGS -c src/process.c -o obj/process.o

echo "[5/5] Compiling main.c ..."
$CC $CFLAGS -c src/main.c -o obj/main.o

echo "Linking ..."
$CC $CFLAGS -o bin/system_monitor \
    obj/main.o obj/cpu.o obj/memory.o obj/disk.o obj/process.o

echo ""
echo "============================================"
echo " Build complete!  Run with:"
echo "   ./bin/system_monitor"
echo "============================================"
