
export VLM_API_KEY=

python code/simulator/main.py \
  --av-mode vlm_live \
  --vlm-model "claude-sonnet-4-5-20250929" \
  --vlm-api-base "https://yinli.one/v1" \
  --vlm-api-key "${VLM_API_KEY}" \
  --vlm-use-top 0 \
  --print-av-action 1 \
  --record-mp4 1
    