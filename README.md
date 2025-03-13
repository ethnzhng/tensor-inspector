# tensor-inspector

```sh
docker run --rm -it \
    --runtime=nvidia \
    --gpus all \
    --shm-size 20gb \
    -v ~/models:/ws/models \
    -v ~/tensor-inspector:/ws/tensor-inspector \
    --workdir /ws \
    --entrypoint /bin/bash \
    vllm/vllm-openai:v0.7.3
```
