curl -X POST "http://localhost:8008/maskrcnn/predict" \
  -H "accept: application/json" \
  -F "file=@input.jpg"