#!/bin/bash

# Test the /chunks/search/ endpoint
ATTEMPTS=3
START_DATE="2025-06-01T00:00:00Z"
NAME_CONTAINS="example"
QUERY_EMBEDDING="[0.1,0.2,0.3]"

for ((i=1; i<=ATTEMPTS; i++)); do
    RESPONSE=$(curl -s -X POST "http://10.10.10.129:8000/chunks/search/" \
         -H "Content-Type: application/json" \
         -d "{\"query_embedding\": $QUERY_EMBEDDING, \"start_date\": \"$START_DATE\", \"name_contains\": \"$NAME_CONTAINS\"}" \
         -w "\nHTTP_STATUS:%{http_code}")
    HTTP_STATUS=$(echo "$RESPONSE" | grep -o 'HTTP_STATUS:[0-9]\+' | cut -d':' -f2)
    RESPONSE=$(echo "$RESPONSE" | sed '/HTTP_STATUS:/d')
    if [ -n "$HTTP_STATUS" ] && [ $HTTP_STATUS -ge 200 ] && [ $HTTP_STATUS -lt 300 ]; then
        echo "JSON Response:"
        echo "$RESPONSE" | jq . 2>/dev/null || echo "$RESPONSE"
        exit 0
    fi
    echo "Attempt $i/$ATTEMPTS failed: HTTP $HTTP_STATUS" >&2
    sleep 2
done

echo "Error: Failed to search chunks after $ATTEMPTS attempts" >&2
echo "$RESPONSE" >&2
exit 1
