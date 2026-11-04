#!/usr/bin/env bash
# Download SoccerMon: https://zenodo.org/records/10033832
# Usage: ./download_soccer_mon.sh [--output DIR] [FILE.zip ...]
# No filenames means download all five archives. Archives are not extracted.
set -euo pipefail

output="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/data/soccermon"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    head -n 4 "$0" | tail -n 3
    exit 0
fi
if [[ "${1:-}" == "--output" ]]; then
    [[ $# -ge 2 ]] || { echo '--output requires a directory' >&2; exit 1; }
    output="$2"
    shift 2
fi
command -v curl >/dev/null || { echo 'Install curl first.' >&2; exit 1; }
command -v md5sum >/dev/null || { echo 'Install md5sum (coreutils) first.' >&2; exit 1; }

declare -A checksums=(
    [objective-TeamA-2020.zip]=e70a71d0a2e39f939fcd13ec5c73411e
    [objective-TeamA-2021.zip]=1c0de36af51c13a4fbde463fa808db0e
    [objective-TeamB-2020.zip]=c68f9d77ebaf042ab3e75a0714587c3c
    [objective-TeamB-2021.zip]=4676436c79d2d0ab3fe1c521a8fd2871
    [subjective.zip]=a3e86aeca611f77c9331a535eae00bf7
)
files=("$@")
if [[ ${#files[@]} -eq 0 ]]; then
    files=(objective-TeamA-2020.zip objective-TeamA-2021.zip objective-TeamB-2020.zip objective-TeamB-2021.zip subjective.zip)
fi
for file in "${files[@]}"; do
    [[ -n "${checksums[$file]:-}" ]] || { echo "Unknown filename: $file" >&2; exit 1; }
done
mkdir -p -- "$output"

verify() {
    printf '%s  %s\n' "${checksums[$1]}" "$2" | md5sum --check --status
}

for file in "${files[@]}"; do
    target="$output/$file"
    partial="$target.part"
    if [[ -f "$target" ]]; then
        echo "Verifying existing $file..."
        if verify "$file" "$target"; then
            echo "Already complete: $file"
            continue
        fi
        echo "Checksum mismatch: $target. Move or remove it, then rerun." >&2
        exit 1
    fi
    echo "Downloading $file (partial downloads resume automatically)..."
    curl --fail --location --retry 5 --retry-delay 5 \
        --connect-timeout 30 --continue-at - --output "$partial" \
        "https://zenodo.org/records/10033832/files/$file?download=1"
    echo "Verifying $file..."
    if ! verify "$file" "$partial"; then
        echo "Checksum mismatch: $partial. Remove it, then rerun." >&2
        exit 1
    fi
    mv -- "$partial" "$target"
done
echo "Downloads verified in: $output"
