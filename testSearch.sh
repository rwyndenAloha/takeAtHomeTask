#!/bin/bash

# Helper function to get the most recent library_id
get_library_id() {
    LIBRARY_ID=$(./testLibrList.sh | jq -r '.[0].id' 2>/dev/null)
    if [ -z "$LIBRARY_ID" ]; then
        echo "Error: Could not retrieve library_id from testLibrList.sh"
        exit 1
    fi
    echo "$LIBRARY_ID"
}

# Get library_id
LIBRARY_ID=$(get_library_id)

# Original curl command with dynamic library_id
curl -X GET "http://localhost:8000/api/v1/libraries/$LIBRARY_ID/search?query=test" \
     -H "Content-Type: application/json" > /dev/null

if [ $? -eq 0 ]; then
    echo "testSearch.sh: Successfully searched library $LIBRARY_ID"
else
    echo "testSearch.sh: Failed to search library $LIBRARY_ID"
    exit 1
fi
