#!/bin/bash
set -euo pipefail

BASE_URL="https://zenodo.org/records/15615521/files"
DATASETS_DIR="$(cd "$(dirname "$0")/.." && pwd)/datasets"

mkdir -p "$DATASETS_DIR/real" "$DATASETS_DIR/g2" "$DATASETS_DIR/scale"

# Returns 0 if every file in FILES array exists under DIR, 1 otherwise.
all_present() {
    local dir="$1"; shift
    for f in "$@"; do
        [[ -f "$dir/$f" ]] || return 1
    done
    return 0
}

# ---------------------------------------------------------------------------
# real_datasets.tar.xz  (2.1 MB) — flat paths inside tar
# ---------------------------------------------------------------------------
REAL_FILES=(
    birch2.txt
    iris.txt
    adult.txt
)

if all_present "$DATASETS_DIR/real" "${REAL_FILES[@]}"; then
    echo "real datasets already present, skipping."
else
    echo "Downloading real_datasets.tar.xz (2.1 MB)..."
    curl -fsSL "$BASE_URL/real_datasets.tar.xz?download=1" \
        | tar -xJf - -C "$DATASETS_DIR/real" "${REAL_FILES[@]}"
    echo "  -> real datasets extracted."
fi

# ---------------------------------------------------------------------------
# g2_datasets.tar.xz  (49.9 MB) — flat paths inside tar
# ---------------------------------------------------------------------------
G2_FILES=(
    g2-4-50.txt
    g2-16-50.txt
    g2-64-50.txt
    g2-128-50.txt
)

if all_present "$DATASETS_DIR/g2" "${G2_FILES[@]}"; then
    echo "g2 datasets already present, skipping."
else
    echo "Downloading g2_datasets.tar.xz (49.9 MB)..."
    curl -fsSL "$BASE_URL/g2_datasets.tar.xz?download=1" \
        | tar -xJf - -C "$DATASETS_DIR/g2" "${G2_FILES[@]}"
    echo "  -> g2 datasets extracted."
fi

# ---------------------------------------------------------------------------
# scale_datasets.tar.xz  (2.2 GB) — flat paths inside tar
# k in {4, 16, 64}, d in {4, 16, 64, 128}, seed=1
# ---------------------------------------------------------------------------
SCALE_FILES=()
for k in 4 16 64; do
    for d in 4 16 64 128; do
        SCALE_FILES+=("SynthNew_${k}_${d}_1.txt")
    done
done

if all_present "$DATASETS_DIR/scale" "${SCALE_FILES[@]}"; then
    echo "scale datasets already present, skipping."
else
    echo "Downloading scale_datasets.tar.xz (2.2 GB — this will take a while)..."
    curl -fsSL "$BASE_URL/scale_datasets.tar.xz?download=1" \
        | tar -xJf - -C "$DATASETS_DIR/scale" "${SCALE_FILES[@]}"
    echo "  -> scale datasets extracted."
fi

# ---------------------------------------------------------------------------
# Label files from cs.joensuu.fi
# ---------------------------------------------------------------------------

# iris.data.txt and b2-gt.txt — single-file downloads
for entry in \
    "real/iris.data.txt|https://cs.joensuu.fi/sipu/datasets/iris.data.txt" \
    "real/b2-gt.txt|https://cs.joensuu.fi/sipu/datasets/b2-gt.txt"
do
    dest="$DATASETS_DIR/${entry%%|*}"
    url="${entry##*|}"
    if [[ -f "$dest" ]]; then
        echo "$(basename "$dest") already present, skipping."
    else
        echo "Downloading $(basename "$url")..."
        curl -fsSL "$url" -o "$dest"
        echo "  -> $dest"
    fi
done

# g2 ground-truth labels — extract only the two files used by get_label_set()
G2_LABEL_FILES=(g2-16-50-gt.txt g2-128-50-gt.txt)
mkdir -p "$DATASETS_DIR/g2/labels"

if all_present "$DATASETS_DIR/g2/labels" "${G2_LABEL_FILES[@]}"; then
    echo "g2 label files already present, skipping."
else
    echo "Downloading g2-gt-txt.zip..."
    TMP_ZIP=$(mktemp --suffix=.zip)
    curl -fsSL "https://cs.joensuu.fi/sipu/datasets/g2-gt-txt.zip" -o "$TMP_ZIP"
    unzip -jo "$TMP_ZIP" "${G2_LABEL_FILES[@]}" -d "$DATASETS_DIR/g2/labels"
    rm -f "$TMP_ZIP"
    echo "  -> g2 label files extracted."
fi

echo "Done."
