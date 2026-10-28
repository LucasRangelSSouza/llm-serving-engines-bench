# Long-context study: cold prefill, prefix-cache hits and multi-turn agent sessions.
# Original harness script, copied unmodified apart from this header. Comments inside may be in Portuguese.
"""Carga de AGENTE com contexto longo (OpenAI-compativel, streaming). Mede o que a bateria curta nao mede: prefill, cache de prefixo e turnos de agente.
uso: python longctx.py --base http://127.0.0.1:18000/v1 --model qwen-abliterated --out DIR --max_ctx 100000 --sessions 1,4,8 --turns 10
Fases (1 linha por requisicao em DIR/longctx.jsonl, campo 'phase'):
  prefill : prompt FRIO de 8K/32K/64K/100K tokens (um por vez) -> TTFT, prefill tok/s, decode tok/s
  prefix  : mesmo prompt + sufixo novo (cache de prefixo quente) -> TTFT quente
  agent   : N sessoes concorrentes; prefixo de sistema compartilhado (12K) + historico que cresce a cada turno -> TTFT/E2E por turno, throughput
Tamanhos reais vem de usage.prompt_tokens (a estimativa por caracteres so dimensiona o texto)."""
import argparse
import asyncio
import json
import random
import time

import httpx

WORDS = ("servidor fila tenant plano cobranca cliente pedido estoque entrega relatorio erro modulo teste codigo deploy banco indice cache token "
         "usuario sessao agente tarefa memoria contexto arquivo funcao classe metodo variavel retorno excecao log metrica painel alerta rede").split()


def text(n_chars, seed):
    r = random.Random(seed)
    out, size = [], 0
    while size < n_chars:
        line = " ".join(r.choice(WORDS) for _ in range(r.randint(8, 16))) + f" #{r.randint(1000, 9999)}\n"
        out.append(line)
        size += len(line)
    return "".join(out)[:n_chars]


async def call(client, args, msgs, max_tokens, phase, meta, out):
    body = {"model": args.model, "messages": msgs, "max_tokens": max_tokens, "temperature": 0.2, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    ttft, usage, err, status, n = None, None, None, None, 0
    try:
        async with client.stream("POST", args.base + "/chat/completions", json=body, headers={"Authorization": "Bearer x"}, timeout=1800) as r:
            status = r.status_code
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    j = json.loads(d)
                except Exception:  # noqa: BLE001
                    continue
                if j.get("usage"):
                    usage = j["usage"]
                ch = j.get("choices") or []
                if ch:
                    dl = ch[0].get("delta", {})
                    if dl.get("content") or dl.get("reasoning") or dl.get("reasoning_content"):
                        n += 1
                        if ttft is None:
                            ttft = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        err = repr(e)[:200]
    total = time.perf_counter() - t0
    pt = (usage or {}).get("prompt_tokens")
    ct = (usage or {}).get("completion_tokens") or n
    cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    rec = {"phase": phase, **meta, "status": status, "error": err, "ttft_s": ttft, "total_s": round(total, 3), "prompt_tokens": pt, "cached_tokens": cached,
           "completion_tokens": ct, "prefill_tok_s": round(pt / ttft, 1) if pt and ttft else None,
           "decode_tok_s": round(ct / (total - ttft), 1) if ttft and total > ttft and ct else None, "t_wall": time.time()}
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--max_ctx", type=int, default=100000)
    ap.add_argument("--sessions", default="1,4,8")
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--sys_tokens", type=int, default=12000)
    a = ap.parse_args()
    import os
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "longctx.jsonl")
    cpt = 3.4  # caracteres por token (estimativa); o real vem do usage
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=64)) as c:
        # warmup curto
        await call(c, a, [{"role": "user", "content": "Responda apenas: ok"}], 8, "warmup", {}, out)
        # 1) prefill frio + 2) prefixo quente
        for ctx in (8000, 32000, 64000, 100000):
            if ctx > a.max_ctx:
                continue
            base = text(int(ctx * cpt), seed=ctx)
            m = [{"role": "user", "content": base + "\n\nResuma em 3 linhas."}]
            r1 = await call(c, a, m, 128, "prefill", {"ctx": ctx}, out)
            m2 = [{"role": "user", "content": base + "\n\nAgora liste 3 riscos tecnicos."}]
            await call(c, a, m2, 128, "prefix", {"ctx": ctx}, out)
            if r1.get("error") or r1.get("status") != 200:
                break
        # 3) sessoes de agente
        sysmsg = {"role": "system", "content": text(int(a.sys_tokens * cpt), seed=7)}
        for ns in [int(x) for x in a.sessions.split(",")]:
            async def session(sid):
                hist = [sysmsg, {"role": "user", "content": "Contexto inicial do repositorio:\n" + text(int(8000 * cpt), seed=100 + sid + ns * 10)}]
                for t in range(a.turns):
                    hist.append({"role": "user", "content": f"Turno {t}: analise o proximo trecho e proponha uma mudanca.\n" + text(int(1500 * cpt), seed=1000 + sid * 50 + t)})
                    rec = await call(c, a, hist, 300, "agent", {"sessions": ns, "sid": sid, "turn": t}, out)
                    if rec.get("error") or rec.get("status") != 200:
                        return
                    hist.append({"role": "assistant", "content": "Entendido. Proposta registrada para o trecho analisado, com a mudanca sugerida e o risco associado."})
                    if (rec.get("prompt_tokens") or 0) > a.max_ctx - 3000:
                        return
            t0 = time.time()
            await asyncio.gather(*[session(i) for i in range(ns)])
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps({"phase": "agent_summary", "sessions": ns, "wall_s": round(time.time() - t0, 1)}) + "\n")
    print("LONGCTX_DONE")


if __name__ == "__main__":
    asyncio.run(main())
