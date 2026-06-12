#!/bin/bash

# Defaults
PROJECT_ROOT="."
DRY_RUN=false
MODE="py:percent"
PREFIX="."
EXCLUDE_PATHS=()

usage() {
    echo "Usage: $0 [PROJECT_ROOT] [--to-ipynb | --to-py] [--prefix PATH] [--exclude PATH] [--dry-run]"
    echo "  --to-py          Convert .ipynb to .py (percent format) [Default]"
    echo "  --to-ipynb       Convert .py to .ipynb"
    echo "  -p, --prefix     Output path prefix (default: .). Use \".\" for same dir."
    echo "  -e, --exclude    Path/pattern to exclude from scanning (can be specified multiple times)"
    echo "  --dry-run        Show actions without executing"
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --to-py) MODE="py:percent"; shift ;;
        --to-ipynb) MODE="notebook"; shift ;;
        -p|--prefix)
            if [[ -n "$2" && "$2" != -* ]]; then
                PREFIX="$2"
                shift 2
            else
                PREFIX="."
                shift 1
            fi
            ;;
        -e|--exclude)
            if [[ -n "$2" && "$2" != -* ]]; then
                EXCLUDE_PATHS+=("$2")
                shift 2
            else
                echo "Error: --exclude requires a value."
                exit 1
            fi
            ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage ;;
        *) PROJECT_ROOT="$1"; shift ;;
    esac
done

# Ensure empty prefix string also becomes "."
PREFIX="${PREFIX:-.}"

if [ "$MODE" == "py:percent" ]; then
    SEARCH_EXT="*.ipynb"
    OUT_EXT=".py"
else
    SEARCH_EXT="*.py"
    OUT_EXT=".ipynb"
fi

# Dynamically construct the find command components for exclusions
FIND_EXCLUDES=()

# First, retain your original rule to skip hidden folders (like .git, .ipynb_checkpoints)
FIND_EXCLUDES+=("-path" '*/.*' "-prune" "-o")

# Append user-defined exclusions from the --exclude flag array
for pattern in "${EXCLUDE_PATHS[@]}"; do
    # Handle if the user passes an absolute path, local path, or generic glob pattern
    FIND_EXCLUDES+=("-path" "*${pattern}*" "-prune" "-o")
done

# Execute the dynamic find construction
find "$PROJECT_ROOT" "${FIND_EXCLUDES[@]}" -name "$SEARCH_EXT" -type f -print | while read -r input_file; do
    dir=$(dirname "$input_file")
    base=$(basename "$input_file" | sed 's/\.[^.]*$//')
    
    # Construct the output path
    output_path="$dir/$PREFIX/${base}${OUT_EXT}"
    cmd="jupytext --to $MODE \"$input_file\" -o \"$output_path\""
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] $cmd"
    else
        echo "Processing: $input_file -> $output_path"
        mkdir -p "$(dirname "$output_path")"
        eval "$cmd"
    fi
done
