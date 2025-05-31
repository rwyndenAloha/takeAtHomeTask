#!/bin/bash

# Helper function to get the most recent library_id
get_library_id() {
    # Run testLibrList.sh and parse the first library_id from JSON
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
curl -X POST "http://10.10.10.129:8000/api/v1/libraries/$LIBRARY_ID/documents" \
     -H "Content-Type: application/json" \
     -d '{"msg_body": "Sample message for testing"}' > /dev/null

if [ $? -eq 0 ]; then
    echo "testMsgBodyParam.sh: Successfully posted document to library $LIBRARY_ID"
else
    echo "testMsgBodyParam.sh: Failed to post document to library $LIBRARY_ID"
    exit 1
fi
