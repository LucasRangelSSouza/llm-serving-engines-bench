"""KV-cache capacity at server start (from the vLLM logs) and long-context behaviour: cold prefill against prefix-cache hits."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import BLUE, GREEN, ORANGE, PURPLE, gpu_name, read, save_fig, save_table, update_summary

st = read("vllm_starts_all.csv")
st["gpu"] = st.gpu.map(gpu_name)
ok = st[st.kv_cache_tokens.notna() & ~st.speculative_decoding].copy()
ok["mib_per_token"] = ok.available_kv_gib * 1024 / ok.kv_cache_tokens
ok["build"] = np.where(ok.quantization == "compressed", "27B AWQ", np.where(ok.log_file.str.contains("fp8"), "27B FP8", np.where(ok.log_file.str.contains("q9|9b|bf16"), "9B BF16", "other")))
tab = ok.groupby(["gpu", "build"]).agg(starts=("kv_cache_tokens", "size"), weights_gib=("model_loading_gib", "median"), cache_gib=("available_kv_gib", "median"), cache_tokens=("kv_cache_tokens", "max"), mib_per_token=("mib_per_token", "median")).reset_index()
save_table(tab.round(3), "kv_cache_by_gpu_and_build", index=False)

failed = st[st.engine_failed_to_start]
fail_tab = failed.groupby("gpu").size().rename("failed_starts")
save_table(fail_tab.to_frame(), "failed_starts_by_gpu")

lc = read("longctx_runs.csv")
pf = lc[lc.phase.isin(["prefill", "prefix"]) & lc.prefill_tok_s.notna()]
piv = pf.pivot_table(index=["config", "ctx"], columns="phase", values="ttft_s", aggfunc="median").reset_index()
piv["speedup"] = piv.prefill / piv.prefix
save_table(piv.round(3), "prefix_cache_speedup", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
b = tab[tab.build.isin(["27B AWQ", "9B BF16", "27B FP8"])]
labels = [f"{g}\n{bld}" for g, bld in zip(b.gpu, b.build)]
axes[0].bar(range(len(b)), b.cache_tokens, color=[BLUE if "AWQ" in x else (GREEN if "9B" in x else ORANGE) for x in labels])
axes[0].set_xticks(range(len(b))); axes[0].set_xticklabels(labels, fontsize=6, rotation=60, ha="right"); axes[0].set_yscale("log")
axes[0].set_ylabel("Tokens of KV cache at server start (all conversations)")
for cfg, col in (("RTX PRO 6000 + NVFP4 27B", PURPLE), ("RTX 5090 + AWQ 27B", BLUE)):
    d = piv[piv.config == cfg]
    axes[1].plot(d.ctx, d.prefill, marker="o", color=col, label=f"{cfg}: cold")
    axes[1].plot(d.ctx, d.prefix, marker="s", ls="--", color=col, label=f"{cfg}: prefix cached")
axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].set_xlabel("Prompt length class (tokens, nominal)"); axes[1].set_ylabel("Time to first token (s)"); axes[1].legend(frameon=False, fontsize=6)
save_fig(fig, "kv_cache_and_prefix_cache.png")

w = ok[(ok.gpu.isin(["RTX 3090", "RTX 4090"])) & (ok.build == "27B AWQ")]
u = ok[(ok.gpu.isin(["RTX 3090", "RTX 4090"])) & (ok.build == "9B BF16")]
update_summary({"awq27b_cache_tokens_24gb_median": w.kv_cache_tokens.median(), "bf16_9b_cache_tokens_24gb_median": u.kv_cache_tokens.median(),
                "awq27b_mib_per_token": w.mib_per_token.median(), "bf16_9b_mib_per_token": u.mib_per_token.median(),
                "failed_starts_total": int(len(failed)), "failed_oom": int(failed.oom_reported.sum()), "starts_total": int(len(st)),
                "best_prefix_speedup": piv.speedup.max()})
print(tab.round(2).head(12))
