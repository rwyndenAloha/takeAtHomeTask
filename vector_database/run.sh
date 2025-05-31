#!/bin/bash
docker build -t vector-db-api .
docker run -p 8000:8000 vector-db-api

echo "API is at http://localhost:8000.  Interactive docs at http://localhost:8000/docs."

