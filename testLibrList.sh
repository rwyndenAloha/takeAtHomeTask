#!/bin/bash
response=$(curl -s -L -w "\n%{http_code}" -X GET "http://10.10.10.129:8000/libraries")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
if [ "$http_code" -eq 200 ]; then
    echo "$body" | jq .  # Pretty-print JSON
else
    echo "Error: HTTP $http_code"
    echo "$body"
fi
