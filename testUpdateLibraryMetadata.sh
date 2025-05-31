#!/bin/bash
library_id="ba75b96a-7dcb-4800-bbb8-d69907e67198"
curl -X PUT "http://10.10.10.129:8000/libraries/$library_id" \
     -H "Content-Type: application/json" \
     -d '{"metadata": {"updated": "true", "category": "test"}}'
