"""Read-only quantitative review of frozen results; writes derived summaries only."""
from pathlib import Path
import json
import sys
import time
import hashlib
import numpy as np
import pandas as pd

S = Path(__file__).resolve().parents[1]
O = Path(__file__).resolve().parent
sys.path.insert(0, str(S / "code"))
from mechanism_base import object_hash


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def out(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(O / (name + ".csv"), index=False, encoding="utf-8-sig")
    return frame


def main():
    results = {f.parent.name: read(f) for f in (S / "runs").glob("*/result.json")}
    receipts = [read(f) for f in (S / "receipts").glob("*.json")]
    manifests = {}
    for job in read(S / "plan.json")["jobs"]:
        name = job["id"]
        m = read(S / "runs" / name / "manifest.json")
        r = read(S / "receipts" / ("runs_" + name + ".json"))
        assert r["transport_verified"] and not r["mismatches"] and r["manifest_sha256"] == object_hash(m)
        assert all((S / "runs" / name / f).stat().st_size == v["bytes"] for f, v in m["files"].items())
        manifests[name] = r["manifest_sha256"]
    audit = {"jobs": len(results), "verified_receipts": len(receipts), "verified_bytes": sum(r["bytes"] for r in receipts),
             "files": sum(r["files"] for r in receipts), "size_and_manifest_recheck_passed": True,
             "prior_sha256_receipts_valid": True, "source_behavior_matches": [], "parameter_freeze_audits": 0,
             "training_stop_preserved": (S.parent / "STOP").exists(), "manifests": manifests}
    for name, r in results.items():
        assert r["status"] == "complete" and not r.get("parameters_updated", False)
        if name.startswith("record-"):
            assert r["source_acceptance_matches"]
            audit["source_behavior_matches"].append(name)
        for a in r.get("frozen_audits", []) + ([r["frozen_audit"]] if "frozen_audit" in r else []):
            assert a["parameters_unchanged"]
            audit["parameter_freeze_audits"] += 1
    (O / "integrity.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    registry = pd.read_csv(S / "tables/run-index.csv")
    difficulty = registry[(registry.module == "difficulty") & (registry.origin == "local")].copy()
    difficulty["tier"] = difficulty.task.str.slice(0, 2)
    out("difficulty-summary", difficulty.groupby("tier").agg(maps=("id", "size"), solved=("original_solved", "sum"),
        median_updates=("updates", "median"), min_updates=("updates", "min"), max_updates=("updates", "max"),
        median_seconds=("training_seconds", "median"), min_seconds=("training_seconds", "min"), max_seconds=("training_seconds", "max")))
    # Write index explicitly: group labels must remain available in the CSV.
    d = difficulty.groupby("tier").agg(maps=("id", "size"), solved=("original_solved", "sum"), median_updates=("updates", "median"),
        min_updates=("updates", "min"), max_updates=("updates", "max"), median_seconds=("training_seconds", "median"),
        min_seconds=("training_seconds", "min"), max_seconds=("training_seconds", "max")).reset_index()
    out("difficulty-summary", d)

    combos = []
    for name, r in results.items():
        if r.get("kind") != "static":
            continue
        b = read(S / "runs" / name / "behaviors.json")
        for v in b:
            key = str(v['brain_phase']) + str(v['head_phase'])
            combos.append({"model": r["model"], "combination": key, "status": v["status"], "steps": v.get("steps"),
                           "solved": v["status"] == "solved"})
    out("four-combination-behavior", combos)
    dec = pd.read_csv(S / "tables/brain-head-decomposition.csv")
    piv = dec.pivot(index="model", columns="contribution", values="mean_margin_change")
    piv["total"] = piv.sum(axis=1)
    out("decomposition-by-model", piv.reset_index())
    neurons = pd.read_parquet(S / "neurons/annotations.parquet")
    neuron_stats = []; anatomical = []
    with np.load(S / "neurons/index.npz") as n:
        groups = n["groups"].copy(); observed = n["observed"].copy(); inputs = n["inputs"].copy()
    for row in difficulty.itertuples():
        before = None
        for phase in ("before", "mid", "after"):
            a = pd.read_parquet(S / "atlas" / f"{row.id}-{phase}.parquet")
            assert np.array_equal(a.bodyId, neurons.bodyId)
            if phase == "before":
                before = a.v_mean.to_numpy().copy()
            dv = a.v_mean.to_numpy() - before
            masks = {"all": np.ones(len(a), dtype=bool), "input": inputs, "observed": observed,
                     "other": ~inputs & ~observed}
            for label, mask in masks.items():
                x = a.loc[mask]
                neuron_stats.append(dict(model=row.id, phase=phase, cells=len(x), population=label,
                    active_cells=int((x.spikes_mean > 0).sum()), spike_probability=float(x.spikes_mean.mean()),
                    mean_voltage=float(x.v_mean.mean()), adaptation=float(x.adapt_mean.mean()),
                    mean_resource=float(x.res_mean.mean()), resource_below_one=int((x.res_mean < 1-1e-6).sum()),
                    mean_abs_voltage_change=float(np.abs(dv[mask]).mean()),
                    voltage_change_p50=float(np.quantile(np.abs(dv[mask]), .5)),
                    voltage_change_p95=float(np.quantile(np.abs(dv[mask]), .95))))
            if phase == "after":
                for col in ("nt", "superclass", "somaNeuromere"):
                    tmp = pd.DataFrame({"label": neurons[col].fillna("unannotated"), "delta": np.abs(dv),
                                        "spike": a.spikes_mean, "adapt": a.adapt_mean, "res": a.res_mean})
                    f = tmp.groupby("label", sort=False).agg(cells=("delta", "size"), mean_abs_voltage_change=("delta", "mean"),
                        spike_probability=("spike", "mean"), adaptation=("adapt", "mean"), resource=("res", "mean")).reset_index()
                    anatomical.extend(f.assign(model=row.id, annotation=col).to_dict("records"))
    out("neuron-weighted-atlas", neuron_stats); out("annotation-functional-summary", anatomical)

    c = pd.read_csv(S / "tables/causal-effects.csv")
    base = {}
    causal_rows = []
    for row in c.to_dict("records"):
        if pd.isna(row["case"]):
            continue
        job = row["job"]; delay = int(row["delay"]); prefix = int(row["prefix"])
        key = (job, delay, prefix)
        if key not in base:
            rows = read(S / "runs" / job / "conditions" / f"d{delay}-p{prefix}-baseline.json")["rows"]
            base[key] = {x["case"]: x for x in rows}
        b = base[key][row["case"]]; q0 = np.asarray(b["decisions"][0]["logits"])
        q1 = np.asarray(json.loads(row["first_logits"]))
        legal = np.asarray(b["decisions"][0]["legal"], dtype=bool)
        target = b["decisions"][0]["action"]; alternatives = np.flatnonzero(legal & (np.arange(4) != target))
        competitor = int(alternatives[np.argmax(q0[alternatives])]) if len(alternatives) else target
        first_margin_delta = float((q1[target]-q1[competitor]) - (q0[target]-q0[competitor]))
        r = {k:row[k] for k in ("job", "model", "condition", "delay", "prefix", "case", "cue", "context", "visible", "status", "baseline_status", "solved_effect")}
        r.update(solved=row["status"] == "solved", baseline_solved=b["status"] == "solved",
                 first_push_correct=row["first_push_correct"] is True, baseline_push=b["first_push_correct"] is True,
                 first_move_margin_delta=first_margin_delta,
                 split="train" if delay in (0,40,160) and prefix in (0,1,4) else "heldout",
                 initial_action_changed=int(np.flatnonzero(legal)[np.argmax(q1[legal])]) != target)
        spec = row["condition"].split("-")
        if job.startswith("components-"):
            r.update(kind="component", window=spec[0], component=spec[1], intervention=spec[2] if len(spec)>2 else spec[1])
        else:
            r.update(kind="group", window=spec[-1] if len(spec)>3 else "control",
                     group="-".join(spec[:2]) if len(spec)>3 else spec[0],
                     side=spec[2] if len(spec)>3 else "control", strength=float(spec[3]) if len(spec)>3 else 0.)
        causal_rows.append(r)
    cc = out("memory-intervention-pairs", causal_rows)
    keys = ["model", "kind", "condition", "visible"]
    summaries = cc.groupby(keys, dropna=False).agg(histories=("case","size"), solved=("solved","sum"), baseline_solved=("baseline_solved","sum"),
        net_change=("solved_effect","sum"), first_push=("first_push_correct","sum"), baseline_push=("baseline_push","sum"),
        mean_first_move_margin_change=("first_move_margin_delta","mean"), changed_initial_action=("initial_action_changed","sum")).reset_index()
    summaries["solution_rate_change"] = summaries.net_change / summaries.histories
    out("memory-intervention-summary", summaries)
    out("static-intervention-summary", c[c.job.str.startswith("intervene-")][["model","group","side","strength","window","baseline_solved","solved","solved_effect"]])
    # Preserve history-stratified paired effects, rather than hiding failures by averaging.
    out("component-history-summary", cc[cc.kind=="component"].groupby(["model","visible","delay","prefix","window","component","intervention"],dropna=False).agg(
        histories=("case","size"), solved=("solved","sum"), baseline_solved=("baseline_solved","sum"), net_change=("solved_effect","sum"),
        first_push=("first_push_correct","sum"), baseline_push=("baseline_push","sum")).reset_index())
    behavior = pd.read_csv(S / "tables/memory-behavior.csv")
    behavior["solved"] = behavior.status.eq("solved")
    out("memory-behavior-summary", behavior.groupby(["model","visible","split"]).agg(histories=("cue","size"),solved=("solved","sum"),
        first_push=("first_push_correct","sum")).reset_index())
    out("memory-delay-summary", behavior.groupby(["model","visible","delay"]).agg(histories=("cue","size"),solved=("solved","sum"),first_push=("first_push_correct","sum")).reset_index())
    geometry = pd.read_csv(S / "tables/memory-geometry.csv")
    out("geometry-summary", geometry.groupby(["model","phase","visible"]).agg(cue_distance=("cue_distance","mean"),
        readout_fraction=("readout_energy_fraction","mean"), contrast=("centered_logit_contrast","mean"),
        min_distance=("cue_distance","min"),max_distance=("cue_distance","max")).reset_index())
    cf = pd.read_csv(S / "tables/cross-family-memory-decoding.csv")
    out("cross-family-memory-summary", cf.groupby(["evaluation_model","heldout_history"]).agg(histories=("cue","size"),correct=("decoded_correct","sum")).reset_index())
    cl = pd.read_csv(S / "tables/cross-family-learning-decoding.csv")
    cl["accuracy"] = cl.correct/cl.states
    out("cross-family-learning-summary", cl.groupby(["phase","representation","target"]).agg(map_families=("model","size"),
        map_mean_accuracy=("accuracy","mean"), map_mean_brier=("brier","mean"), map_mean_majority_baseline=("majority_baseline_accuracy","mean")).reset_index())

    flux = np.load(S / "runs/pathways/actual-direct-transmission.npz")
    flux_info = {k: {"shape":list(flux[k].shape), "dtype":str(flux[k].dtype)} for k in flux.files}
    (O / "flux-schema.json").write_text(json.dumps(flux_info,indent=2),encoding="utf8")
    print(json.dumps({"review_complete":True,"verified_jobs":len(results),"causal_pairs":len(cc),"neuron_summaries":len(neuron_stats),"flux_schema":flux_info},ensure_ascii=False))


if __name__ == "__main__":
    main()
