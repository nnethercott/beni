curl http://127.0.0.1:5001/ \
  -H "Content-Type: application/json" \
  -d '{
      "model": "beni1.6b",
      "messages": [
        {
          "role": "system",
          "content": "You are a helpful assistant."
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "Describe this image in detail."
            },
            {
              "type": "image_url",
              "image_url": {
                "url": "https://hips.hearstapps.com/hmg-prod/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg?crop=0.752xw:1.00xh;0.175xw,0&resize=1200:*"
              }
            }
          ]
        }
      ],
      "max_tokens": 128,
      "temperature": 0.7,
      "top_k": 64,
      "do_sample": false
    }'
