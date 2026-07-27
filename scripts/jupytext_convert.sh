#!/bin/bash

# Defaults
PROJECT_ROOT="."
DRY_RUN=false
MODE="py:percent"
PREFIX="."
DEST=""
MAINTAIN_STRUCTURE=false
EXCLUDE_PATHS=()

usage() {
    echo "Usage: $0 [PROJECT_ROOT] [--to-ipynb | --to-py] [--destination PATH] [--prefix PATH] [--exclude PATH] [--maintain-structure] [--dry-run]"
    echo "  --to-py              Convert .ipynb to .py (percent format) [Default]"
    echo "  --to-ipynb           Convert .py to .ipynb"
    echo "  -d, --destination    Destination folder for converted output files"
    echo "  -p, --prefix         Optional subpath under destination (default: .). Use \".\" for same dir."
    echo "  -e, --exclude        Path/pattern to exclude from scanning (can be specified multiple times)"
    echo "  --maintain-structure Maintain source directory structure under destination"
    echo "  --dry-run            Show actions without executing"
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --to-py) MODE="py:percent"; shift ;;
        --to-ipynb) MODE="notebook"; shift ;;
        -d|--destination)
            if [[ -n "$2" && "$2" != -* ]]; then
                DEST="$2"
                shift 2
            else
                echo "Error: --destination requires a value."
                exit 1
            fi
            ;;
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
        --maintain-structure)
            MAINTAIN_STRUCTURE=true
            shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage ;;
        *) PROJECT_ROOT="$1"; shift ;;
    esac
done

# Ensure empty prefix string also becomes "."
PREFIX="${PREFIX:-.}"

# Remove trailing "/" from PROJECT_ROOT, PREFIX and DEST if any
PROJECT_ROOT="${PROJECT_ROOT%/}"
PREFIX="${PREFIX%/}"
DEST="${DEST%/}"

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

    if [ -n "$DEST" ] && [ "$DEST" != "." ]; then
        if [ "$MAINTAIN_STRUCTURE" = true ]; then
            rel_dir="${dir#"$PROJECT_ROOT"}"
            rel_dir="${rel_dir#/}"

            output_dir="$DEST"
            if [ "$PREFIX" != "." ]; then
                output_dir="$output_dir/$PREFIX"
            fi
            if [ -n "$rel_dir" ]; then
                output_dir="$output_dir/$rel_dir"
            fi
        else
            output_dir="$DEST"
            if [ "$PREFIX" != "." ]; then
                output_dir="$output_dir/$PREFIX"
            fi
        fi
    else
        output_dir="$dir/$PREFIX"
    fi

    output_path="$output_dir/${base}${OUT_EXT}"
    cmd="jupytext --to $MODE \"$input_file\" -o \"$output_path\""
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] $cmd"
    else
        echo "Processing: $input_file -> $output_path"
        mkdir -p "$(dirname "$output_path")"
        eval "$cmd"
    fi
done
