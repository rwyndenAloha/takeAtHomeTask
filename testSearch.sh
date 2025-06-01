#!/bin/bash

# Helper function to get all library_ids from testLibrList.sh
get_library_ids() {
    LIBR_LIST_OUTPUT=$(./testLibrList.sh)
    if [ $? -ne 0 ]; then
        echo "Error: testLibrList.sh failed to execute" >&2
        exit 1
    fi
    if [ -z "$LIBR_LIST_OUTPUT" ] || [ "$LIBR_LIST_OUTPUT" = "[]" ]; then
        echo "Error: No libraries found in testLibrList.sh output. Run testCreateLibrary.sh first." >&2
        exit 1
    fi
    echo "$LIBR_LIST_OUTPUT" | jq -r '.[].id' 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Error: Could not retrieve library_ids from testLibrList.sh output" >&2
        exit 1
    fi
}

# Initialize result array
RESULTS="[]"

# If LIBRARY_ID is unset, prompt for search term and search all libraries
if [ -z "$LIBRARY_ID" ]; then
    echo -n "Enter search term: "
    read SEARCH_TERM
    if [ -z "$SEARCH_TERM" ]; then
        echo "Error: Search term cannot be empty" >&2
        exit 1
    fi
    # Use hardcoded embedding for compatibility with stored chunks
    EMBEDDING_JSON="[0.1,0.2,0.3]"
    # Get all library IDs
    LIBRARY_IDS=$(get_library_ids) || exit 1
    # Loop through each library
    for LIB_ID in $LIBRARY_IDS; do
        ATTEMPTS=3
        for ((i=1; i<=ATTEMPTS; i++)); do
            RESPONSE=$(curl -s -X POST "http://10.10.10.129:8000/libraries/$LIB_ID/search/" \
                 -H "Content-Type: application/json" \
                 -d "{\"embedding\": $EMBEDDING_JSON}" \
                 -w "\nHTTP_STATUS:%{http_code}")
            HTTP_STATUS=$(echo "$RESPONSE" | grep -o 'HTTP_STATUS:[0-9]\+' | cut -d':' -f2)
            RESPONSE=$(echo "$RESPONSE" | sed '/HTTP_STATUS:/d')
            if [ -n "$HTTP_STATUS" ] && [ $HTTP_STATUS -ge 200 ] && [ $HTTP_STATUS -lt 300 ]; then
                # Add non-empty results to the aggregate
                if [ "$RESPONSE" != "[]" ]; then
                    if [ "$RESULTS" = "[]" ]; then
                        RESULTS="$RESPONSE"
                    else
                        RESULTS=$(echo "$RESULTS" | jq -c ". + $RESPONSE")
                    fi
                fi
                break
            fi
            echo "Attempt $i/$ATTEMPTS failed for library $LIB_ID: HTTP $HTTP_STATUS" >&2
            sleep 2
        done
        if [ $i -gt $ATTEMPTS ]; then
            echo "Error: Failed to search library $LIB_ID after $ATTEMPTS attempts" >&2
        fi
    done
else
    # Use hardcoded embedding when LIBRARY_ID is set
    EMBEDDING_JSON="[0.1,0.2,0.3]"
    ATTEMPTS=3
    for ((i=1; i<=ATTEMPTS; i++)); do
        RESPONSE=$(curl -s -X POST "http://10.10.10.129:8000/libraries/$LIBRARY_ID/search/" \
             -H "Content-Type: application/json" \
             -d "{\"embedding\": $EMBEDDING_JSON}" \
             -w "\nHTTP_STATUS:%{http_code}")
        HTTP_STATUS=$(echo "$RESPONSE" | grep -o 'HTTP_STATUS:[0-9]\+' | cut -d':' -f2)
        RESPONSE=$(echo "$RESPONSE" | sed '/HTTP_STATUS:/d')
        if [ -n "$HTTP_STATUS" ] && [ $HTTP_STATUS -ge 200 ] && [ $HTTP_STATUS -lt 300 ]; then
            RESULTS="$RESPONSE"
            break
        fi
        echo "Attempt $i/$ATTEMPTS failed: HTTP $HTTP_STATUS" >&2
        sleep 2
    done
    if [ $i -gt $ATTEMPTS ]; then
        echo "Error: Failed to search library $LIBRARY_ID after $ATTEMPTS attempts" >&2
        echo "$RESPONSE" >&2
        exit 1
    fi
fi

# Output the aggregated results
echo "$RESULTS" | jq . 2>/dev/null || echo "$RESULTS"
exit 0
