#!/bin/bash

curl -X POST http://10.10.10.129:8000/api/v1/chunks/search/ \
  -H "Content-Type: application/json" \
  -d '{"query_embedding":[0.1,0.2,0.3],"start_date":"2025-06-01T00:00:00Z","name_contains":"example"}'

