#!/bin/bash

LIBRARY_ID="fb9c4765-07bc-450c-b784-499d8db4c2ad"
DOCUMENT_ID="cdcb7d99-687b-4f3e-9bb0-a5b945e1986d"

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
