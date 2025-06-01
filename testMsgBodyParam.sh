#!/bin/bash

# Helper function to get the most recent library_id
get_library_id() {
    # Run testLibrList.sh and capture output
    LIBR_LIST_OUTPUT=$(./testLibrList.sh)
    if [ $? -ne 0 ]; then
        echo "Error: testLibrList.sh failed to execute" >&2
        exit 1
    fi
    # Log output for debugging to stderr
    echo "testLibrList.sh output: $LIBR_LIST_OUTPUT" >&2
    # Check if output is empty or invalid
    if [ -z "$LIBR_LIST_OUTPUT" ] || [ "$LIBR_LIST_OUTPUT" = "[]" ]; then
        echo "Error: No libraries found in testLibrList.sh output. Run testCreateLibrary.sh first." >&2
        exit 1
    fi
    # Parse the first library_id
    LIBRARY_ID=$(echo "$LIBR_LIST_OUTPUT" | jq -r '.[0].id' 2>/dev/null)
    if [ -z "$LIBRARY_ID" ] || [ "$LIBRARY_ID" = "null" ]; then
        echo "Error: Could not retrieve valid library_id from testLibrList.sh output" >&2
        exit 1
    fi
    echo "$LIBRARY_ID"
}

# Get library_id and exit if it fails
LIBRARY_ID=$(get_library_id) || exit 1

# Run curl with retries (3 attempts)
ATTEMPTS=3
for ((i=1; i<=ATTEMPTS; i++)); do
    RESPONSE=$(curl -X POST "http://10.10.10.129:8000/libraries/$LIBRARY_ID/documents/" \
         -H "Content-Type: application/json" \
         -d '{"text": "Sample message for testing", "metadata": {"source": "example"}}' \
         -w "\nHTTP_STATUS:%{http_code}" 2>&1)

    # Extract HTTP status code
    HTTP_STATUS=$(echo "$RESPONSE" | grep -o 'HTTP_STATUS:[0-9]\+' | cut -d':' -f2)

    # Remove HTTP status line from response
    RESPONSE=$(echo "$RESPONSE" | sed '/HTTP_STATUS:/d')

    # Check if curl was successful (HTTP 2xx status code)
    if [ -n "$HTTP_STATUS" ] && [ $HTTP_STATUS -ge 200 ] && [ $HTTP_STATUS -lt 300 ]; then
        echo "JSON Response:"
        echo "$RESPONSE" | jq . 2>/dev/null || echo "$RESPONSE"  # Pretty-print if valid JSON
        echo "testMsgBodyParam.sh: Successfully posted document to library $LIBRARY_ID"
        exit 0
    fi

    echo "Attempt $i/$ATTEMPTS failed. Retrying..." >&2
    sleep 2
done

# If all attempts fail
echo "Error Response:"
echo "$RESPONSE"
echo "HTTP Status: ${HTTP_STATUS:-Unknown}"
echo "testMsgBodyParam.sh: Failed to post document to library $LIBRARY_ID"
exit 1
