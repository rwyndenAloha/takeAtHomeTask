#!/bin/bash
curl -X POST "http://10.10.10.129:8000/libraries/" \
     -H "Content-Type: application/json" \
     -d '{
           "documents": [
             {
               "chunks": [
                 {
                   "text": "Sample text",
                   "embedding": [0.1, 0.2, 0.3],
                   "metadata": {"source": "example"}
                 }
               ],
               "metadata": {"doc": "test"}
             }
           ],
           "metadata": {"library": "test-lib"}
         }'

