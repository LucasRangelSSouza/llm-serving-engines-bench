#!/bin/bash
# Starts the 27B server with explicit batch, context and memory limits.
# Original harness script, copied unmodified apart from this header. Comments inside may be in Portuguese.
# uso: serve27.sh <variante> [max_seqs] [gpu_util] [max_len]   variantes: fp8_dflash | fp8_plain | nvfp4_dflash | nvfp4_plain | bf16_plain
# Sobe o 27B abliterado (nosso modelo) em :18000 sem api-key (so loopback), espera ficar saudavel e imprime o tempo de carga.
V=${1:-fp8_dflash}; SEQS=${2:-32}; UTIL=${3:-0.85}; LEN=${4:-32768}
. /venv/main/bin/activate
pkill -f "vllm serve" 2>/dev/null; sleep 5
case "$V" in
  fp8_*)   DIR=/workspace/models/bf16;   Q="--quantization fp8" ;;
  nvfp4_*) DIR=/workspace/models/target; Q="" ;;
  bf16_*)  DIR=/workspace/models/bf16;   Q="" ;;
esac
SPEC=""
case "$V" in
  *_dflash) SPEC="--speculative-config {\"method\":\"dflash\",\"model\":\"/workspace/models/drafter\",\"num_speculative_tokens\":7}" ;;
esac
LOG=/workspace/eco/out/vllm_${V}.log
T0=$(date +%s)
nohup vllm serve $DIR --served-model-name qwen-abliterated --host 127.0.0.1 --port 18000 \
  --gpu-memory-utilization $UTIL --max-model-len $LEN --max-num-seqs $SEQS --kv-cache-dtype auto $Q \
  --trust-remote-code --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 $SPEC > $LOG 2>&1 &
for i in $(seq 1 240); do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:18000/health)
  [ "$code" = 200 ] && break
  if ! pgrep -f "vllm serve" >/dev/null; then echo "vllm morreu"; tail -15 $LOG; exit 1; fi
  sleep 5
done
echo "variant=$V health=$code load_s=$(( $(date +%s) - T0 ))"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
