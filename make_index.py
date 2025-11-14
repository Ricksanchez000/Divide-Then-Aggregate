# make_index.py
import os, json, torch, glob

model_dir = "/home/t653z659/.cache/huggingface/hub/models--dongsheng--DTA_llama2_7b/snapshots/68b8ebcd2cd6f6f34d6e6c88b1a9e8e081bffae2"

shards = sorted(glob.glob(os.path.join(model_dir, "pytorch_model-*-of-*.bin")))
assert shards, "No shard files found"

weight_map = {}
total_size = 0

def tensor_nbytes(t):
    # 兼容旧权重里偶尔出现的 half/float 等
    return t.numel() * (torch.finfo(t.dtype).bits // 8)

for shard in shards:
    print(f"[index] scanning {os.path.basename(shard)}")
    state = torch.load(shard, map_location="cpu")
    # 典型是直接就是 state_dict；若是 {'state_dict': ...} 结构，解一层
    if isinstance(state, dict) and "state_dict" in state and len(state) <= 2:
        state = state["state_dict"]
    for k, v in state.items():
        # 只记录张量；某些文件可能混有标量/列表
        if torch.is_tensor(v):
            weight_map[k] = os.path.basename(shard)
            total_size += tensor_nbytes(v)
        # 有些键是张量片段（.index/.param），忽略非张量

index = {
    "metadata": {"total_size": int(total_size)},
    "weight_map": weight_map,
}
out = os.path.join(model_dir, "pytorch_model.bin.index.json")
with open(out, "w") as f:
    json.dump(index, f)
print("Wrote:", out, " with", len(weight_map), "params")
