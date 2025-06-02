#!/bin/bash
docker build -t vector-db-api .
mkdir -p ./data # local data directory to persist storage of json data
docker run -v $(pwd)/data:/data -p 8000:8000 vector-db-api

echo "API is at http://localhost:8000.  Interactive docs at http://localhost:8000/docs."

