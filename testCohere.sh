#!/bin/bash

# Helper function to get all library_ids
get_library_ids() {
    LIBRARY_IDS=$(./testLibrList.sh | jq -r '.[]?.id' 2>/dev/null)
    if [ -z "$LIBRARY_IDS" ]; then
        echo "Error: Could not retrieve library_ids from testLibrList.sh"
        exit 1
    fi
    echo "$LIBRARY_IDS"
}

# Get all library_ids
LIBRARY_IDS=$(get_library_ids)

# Check if python3 and embed.py exist
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi
if [ ! -f "embed.py" ]; then
    echo "Error: embed.py not found"
    exit 1
fi

# Iterate over each library_id
for LIBRARY_ID in $LIBRARY_IDS; do
    echo "Processing library $LIBRARY_ID"
    # Modify embed.py input to use the library_id (assumes embed.py accepts library_id as env var or arg)
    export LIBRARY_ID
    python3 embed.py > /dev/null
    if [ $? -eq 0 ]; then
        echo "testCohere.sh: Successfully generated embeddings for library $LIBRARY_ID"
    else
        echo "testCohere.sh: Failed to generate embeddings for library $LIBRARY_ID"
    fi
done

