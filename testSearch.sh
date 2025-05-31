#!/bin/bash
curl -X POST "http://10.10.10.129:8000/libraries/ba75b96a-7dcb-4800-bbb8-d69907e67198/search/" \
     -H "Content-Type: application/json" \
     -d '{"embedding": [0.1, 0.2, 0.3], "k": [2]}'
