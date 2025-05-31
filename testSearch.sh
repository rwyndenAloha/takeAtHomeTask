#!/bin/bash
curl -X POST "http://10.10.10.129:8000/libraries/f3f08fd0-6f7a-4946-8338-aba7a9f8e239/search/" \
     -H "Content-Type: application/json" \
     -d '{"embedding": [0.1, 0.2, 0.3], "k": [2]}'
