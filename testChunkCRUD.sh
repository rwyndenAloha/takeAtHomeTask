#!/bin/bash

LIBRARY_ID="fb9c4765-07bc-450c-b784-499d8db4c2ad"
DOCUMENT_ID="f435fcf7-c122-44f4-81fc-64bc20fec171"
CHUNK_ID="b4032000-be36-4935-b620-7f73c47c1e07"

# Create Chunk
echo "Creating chunk"
RESPONSE=$(curl -s -X POST "http://10.10.10.129:8000/libraries/$LIBRARY_ID/documents/$DOCUMENT_ID/chunks/" \
     -H "Content-Type: application/json" \
     -d '{"text": "New chunk", "embedding": [0.4, 0.5, 0.6], "metadata": {"source": "new"}}' \
     -w "\nHTTP_STATUS:%{http_code}")
HTTP_STATUS=$(echo "$RESPONSE" | grep -o 'HTTP_STATUS:[0-9]\+' | cut -d':' -f2)
RESPONSE=$(echo "$RESPONSE" | sed '/HTTP_STATUS:/d')
NEW_CHUNK_ID=$(echo "$RESPONSE" | jq -r '.id' 2>/dev/null)
if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "JSON Response:"
    echo "$RESPONSE" | jq .
else
    echo "Error Response:"
    echo "$RESPONSE"
    exit 1
fi

# Read Chunk
echo "Reading chunk $NEW_CHUNK_ID"
RESPONSE=$(curl -s -X GET "http://10.10.10.129:8000/libraries/$LIBRARY_ID/documents/$DOCUMENT_ID/chunks/$NEW_CHUNK_ID" \
     -w "\nHTTP_STATUS:%{http_code}")
HTTP_STATUS=$(echo "$RESPONSE" | grep -o 'HTTP_STATUS:[0-9]\+' | cut -d':' -f2)
RESPONSE=$(echo "$RESPONSE" | sed '/HTTP_STATUS:/d')
if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "JSON Response:"
    echo "$RESPONSE" | jq .
else
    echo "Error Response:"
    echo "$RESPONSE"
    exit 1
fi

# Update Chunk
echo "Updating chunk $NEW_CHUNK_ID"
RESPONSE=$(curl -s -X PUT "http://10.10.10.129:8000/libraries/$LIBRARY_ID/documents/$DOCUMENT_ID/chunks/$NEW_CHUNK_ID" \
     -H "Content-Type: application/json" \
     -d '{"text": "Updated chunk", "embedding": [0.7, 0.8, 0.9]}' \
     -w "\nHTTP_STATUS:%{http_code}")
HTTP_STATUS=$(echo "$RESPONSE" | grep -o 'HTTP_STATUS:[0-9]\+' | cut -d':' -f2)
RESPONSE=$(echo "$RESPONSE" | sed '/HTTP_STATUS:/d')
if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "JSON Response:"
    echo "$RESPONSE" | jq .
else
    echo "Error Response:"
    echo "$RESPONSE"
    exit 1
fi

# Delete Chunk
echo "Deleting chunk $NEW_CHUNK_ID"
RESPONSE=$(curl -s -X DELETE "http://10.10.10.129:8000/libraries/$LIBRARY_ID/documents/$DOCUMENT_ID/chunks/$NEW_CHUNK_ID" \
     -w "\nHTTP_STATUS:%{http_code}")
HTTP_STATUS=$(echo "$RESPONSE" | grep -o 'HTTP_STATUS:[0-9]\+' | cut -d':' -f2)
RESPONSE=$(echo "$RESPONSE" | sed '/HTTP_STATUS:/d')
if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "JSON Response:"
    echo "$RESPONSE" | jq .
else
    echo "Error Response:"
    echo "$RESPONSE"
    exit 1
fi
