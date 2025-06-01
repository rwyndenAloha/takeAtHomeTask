#!/bin/bash

# Run testLibrList.sh and capture output
LIBRARIES=$(./testLibrList.sh)

# Extract the first library_id and document_id from a library with documents
LIBRARY_ID=$(echo "$LIBRARIES" | jq -r '.[] | select(.documents | length > 0) | .id' | head -n 1)
DOCUMENT_ID=$(echo "$LIBRARIES" | jq -r '.[] | select(.documents | length > 0) | .documents[0].id' | head -n 1)

if [ -z "$LIBRARY_ID" ] || [ "$LIBRARY_ID" = "null" ]; then
    echo "Error: No libraries found or invalid library_id" >&2
    exit 1
fi

if [ -z "$DOCUMENT_ID" ] || [ "$DOCUMENT_ID" = "null" ]; then
    echo "Error: No documents found or invalid document_id" >&2
    exit 1
fi

echo "Testing chunk CRUD operations for library $LIBRARY_ID, document $DOCUMENT_ID"

# Create Chunk
echo "Creating chunk"
RESPONSE=$(curl -s -X POST "http://10.10.10.129:8000/libraries/$LIBRARY_ID/documents/$DOCUMENT_ID/chunks/" \
     -H "Content-Type: application/json" \
     -d '{"text": "New chunk", "embedding": [0.4, 0.5, 0.6], "metadata": {"source": "new"}}' \
     -w "\nHTTP_STATUS:%{http_code}")
HTTP_STATUS=$(echo "$RESPONSE" | grep -o 'HTTP_STATUS:[0-9]\+' | cut -d':' -f2)
RESPONSE=$(echo "$RESPONSE" | sed '/HTTP_STATUS:/d')
NEW_CHUNK_ID=$(echo "$RESPONSE" | jq -r '.id' 2>/dev/null)
if [ "$HTTP_STATUS" -eq 200 ] && [ -n "$NEW_CHUNK_ID" ] && [ "$NEW_CHUNK_ID" != "null" ]; then
    echo "JSON Response:"
    echo "$RESPONSE" | jq .
else
    echo "Error Response:"
    echo "$RESPONSE"
    echo "testChunkCRUD.sh: Failed to create chunk"
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
    echo "testChunkCRUD.sh: Failed to read chunk $NEW_CHUNK_ID"
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
    echo "testChunkCRUD.sh: Failed to update chunk $NEW_CHUNK_ID"
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
    echo "testChunkCRUD.sh: Failed to delete chunk $NEW_CHUNK_ID"
    exit 1
fi
