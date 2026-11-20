"""CPU-only idle tracking, independent of physical GPU indices."""
def update_idle(gpus, busy_uuids, since, now, required=600):
    present = {g["uuid"] for g in gpus}
    for uuid in set(since) - present:
        since.pop(uuid)
    for g in gpus:
        uuid = g["uuid"]
        idle = uuid not in busy_uuids and int(g["memory_mib"]) <= 512 and int(g["utilization"]) <= 5
        if idle:
            since.setdefault(uuid, now)
        else:
            since.pop(uuid, None)
    eligible = [g for g in gpus if g["uuid"] in since and now - since[g["uuid"]] >= required]
    eligible.sort(key=lambda g: (since[g["uuid"]], int(g["index"])))
    return eligible[:4] if len(eligible) >= 4 else []
