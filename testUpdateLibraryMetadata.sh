#!/bin/bash

# Run testLibrList.sh and capture output
LIBRARIES=$(./testLibrList.sh)

# Extract the first library_id using jq
LIBRARY_ID=$(echo "$LIBRARIES" | jq -r '.[0].id' 2>/dev/null)

if [ -z "$LIBRARY_ID" ] || [ "$LIBRARY_ID" = "null" ]; then
    echo "Error: No libraries found or invalid library_id" >&2
    exit 1
fi

echo "Updating metadata for library $LIBRARY_ID"

ATTEMPTS=3
for ((i=1; i<=ATTEMPTS; i++)); do
    RESPONSE=$(curl -v -X PUT "http://10.10.10.129:8000/libraries/$LIBRARY_ID" \
         -H "Content-Type: application/json" \
         -d '{"name": "Updated Library Name 4"}' \
         -w "\nHTTP_STATUS:%{http_code}" 2>&1)
    HTTP_STATUS=$(echo "$RESPONSE" | grep -o 'HTTP_STATUS:[0-9]\+' | cut -d':' -f2)
    RESPONSE=$(echo "$RESPONSE" | sed '/HTTP_STATUS:/d')
    if [ -n "$HTTP_STATUS" ] && [ $HTTP_STATUS -eq 200 ]; then
        echo "JSON Response:"
        echo "$RESPONSE" | jq . 2>/dev/null || echo "$RESPONSE"
        echo "testUpdateLibraryMetadata.sh: Successfully updated metadata for library $LIBRARY_ID"
        exit 0
    fi
    echo "Attempt $i/$ATTEMPTS failed. Retrying..." >&2
    sleep 2
done

echo "Error Response:"
echo "$RESPONSE"
echo "HTTP Status: ${HTTP_STATUS:-Unknown}"
echo "testUpdateLibraryMetadata.sh: Failed to update metadata for library $LIBRARY_ID"
exit 1
