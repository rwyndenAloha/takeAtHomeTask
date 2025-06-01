#!/bin/bash

# Helper function to get all library_ids
get_library_ids() {
    LIBR_LIST_OUTPUT=$(./testLibrList.sh)
    if [ $? -ne 0 ]; then
        echo "Error: testLibrList.sh failed to execute" >&2
        exit 1
    fi
    echo "testLibrList.sh output: $LIBR_LIST_OUTPUT" >&2
    if [ -z "$LIBR_LIST_OUTPUT" ] || [ "$LIBR_LIST_OUTPUT" = "[]" ]; then
        echo "Error: No libraries found in testLibrList.sh output. Run testCreateLibrary.sh first." >&2
        exit 1
    fi
    LIBRARY_IDS=$(echo "$LIBR_LIST_OUTPUT" | jq -r '.[]?.id' 2>/dev/null)
    if [ -z "$LIBRARY_IDS" ]; then
        echo "Error: Could not retrieve library_ids from testLibrList.sh output" >&2
        exit 1
    fi
    echo "$LIBRARY_IDS"
}

# Get all library_ids
LIBRARY_IDS=$(get_library_ids) || exit 1

# Check dependencies
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found" >&2
    exit 1
fi
if [ ! -f "embed.py" ]; then
    echo "Error: embed.py not found" >&2
    exit 1
fi
if [ -z "$COHERE_API_KEY" ]; then
    echo "Error: COHERE_API_KEY environment variable not set" >&2
    exit 1
fi

# Iterate over each library_id
for LIBRARY_ID in $LIBRARY_IDS; do
    echo "Processing library $LIBRARY_ID" >&2
    export LIBRARY_ID
    RESPONSE=$(python3 embed.py 2>&1)
    if [ $? -eq 0 ]; then
        echo "JSON Response:"
        echo "$RESPONSE" | jq . 2>/dev/null || echo "$RESPONSE"
        echo "testCohere.sh: Successfully generated embeddings for library $LIBRARY_ID"
    else
        echo "Error Response:"
        echo "$RESPONSE"
        echo "testCohere.sh: Failed to generate embeddings for library $LIBRARY_ID"
    fi
done
