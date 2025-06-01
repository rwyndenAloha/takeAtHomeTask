#!/bin/bash

# Clear previous embedded_output.json
: > embedded_output.json

# Run testLibrList.sh to get libraries
LIBRARIES=$(./testLibrList.sh)
if [ $? -ne 0 ]; then
    echo "Error: testLibrList.sh failed to execute" >&2
    exit 1
fi
echo "testLibrList.sh output: $LIBRARIES"

# Check if libraries exist
if [ -z "$LIBRARIES" ] || [ "$LIBRARIES" = "[]" ]; then
    echo "Error: No libraries found in testLibrList.sh output" >&2
    exit 1
fi

# Process each library
for LIBRARY_ID in $(echo "$LIBRARIES" | jq -r '.[].id'); do
    echo "Processing library $LIBRARY_ID"
    RESPONSE=$(LIBRARY_ID=$LIBRARY_ID ./embed.py 2>&1)
    if [ $? -eq 0 ]; then
        # Read the entire embedded_output.json
        if [ -s embedded_output.json ]; then
            JSON_OUTPUT=$(cat embedded_output.json)
            echo "JSON Response:"
            echo "$JSON_OUTPUT" | jq . 2>/dev/null || echo "$JSON_OUTPUT"
        else
            echo "Warning: embedded_output.json is empty" >&2
        fi
        echo "testCohere.sh: Successfully generated embeddings for library $LIBRARY_ID"
    else
        echo "Error Response:"
        echo "$RESPONSE"
        echo "testCohere.sh: Failed to generate embeddings for library $LIBRARY_ID"
    fi
done
