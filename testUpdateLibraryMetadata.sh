#!/bin/bash
library_id="f3f08fd0-6f7a-4946-8338-aba7a9f8e239"
curl -X PUT "http://10.10.10.129:8000/libraries/$library_id" \
     -H "Content-Type: application/json" \
     -d '{"metadata": {"updated": "true", "category": "test"}}'
