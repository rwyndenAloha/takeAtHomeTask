#!/bin/bash

LIBRARY_ID="fb9c4765-07bc-450c-b784-499d8db4c2ad"

RESPONSE=$(curl -s -X DELETE "http://10.10.10.129:8000/libraries/$LIBRARY_ID" \
     -w "\nHTTP_STATUS:%{http_code}")
HTTP_STATUS=$(echo "$RESPONSE" | grep -o 'HTTP_STATUS:[0-9]\+' | cut -d':' -f2)
RESPONSE=$(echo "$RESPONSE" | sed '/HTTP_STATUS:/d')

if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "JSON Response:"
    echo "$RESPONSE" | jq . 2>/dev/null || echo "$RESPONSE"
    echo "testDeleteLibrary.sh: Successfully deleted library $LIBRARY_ID"
else
    echo "Error Response:"
    echo "$RESPONSE"
    echo "HTTP Status: $HTTP_STATUS"
    echo "testDeleteLibrary.sh: Failed to delete library $LIBRARY_ID"
    exit 1
fi
