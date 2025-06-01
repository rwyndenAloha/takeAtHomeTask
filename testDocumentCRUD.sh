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

echo "Testing document CRUD operations for library $LIBRARY_ID, document $DOCUMENT_ID"

# Read Document
echo "Reading document $DOCUMENT_ID"
RESPONSE=$(curl -s -X GET "http://10.10.10.129:8000/libraries/$LIBRARY_ID/documents/$DOCUMENT_ID" \
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

# Update Document
echo "Updating document $DOCUMENT_ID"
RESPONSE=$(curl -s -X PUT "http://10.10.10.129:8000/libraries/$LIBRARY_ID/documents/$DOCUMENT_ID" \
     -H "Content-Type: application/json" \
     -d '{"source": "updated_example"}' \
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

# Delete Document
echo "Deleting document $DOCUMENT_ID"
RESPONSE=$(curl -s -X DELETE "http://10.10.10.129:8000/libraries/$LIBRARY_ID/documents/$DOCUMENT_ID" \
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
