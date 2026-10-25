"""Additional descriptive checks on saved trajectories, with a persistence baseline."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

S = Path(__file__).resolve().parents[1]
O = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text(encoding="utf8"))
neurons = pd.read_parquet(S / "neurons/annotations.parquet")
models = [r for r in read(S / "models.json") if r["kind"] == "memory"]
fit_rows = []; cell_rows = []; flux_rows = []
scores = np.load(S / "runs/discover-memory/neuron-scores.npz")
with np.load(S / "runs/discover-memory/discovery-scale.npz") as f:
    scale = f["scale"]

for model in models:
    name = model["id"]; folder = S / "runs" / ("record-" + name)
    with np.load(S / "tables" / (name + "-shared-pca.npz")) as pca:
        mean, std, basis = pca["mean"], pca["scale"], pca["basis"]
    episodes = []; resource_min = 1.; phase_feature_deltas = []
    indices = read(folder / "matched-index.json")
    before = {(r["delay"],r["prefix"]):r for r in indices if r["phase"] == "before"}
    for entry in indices:
        if entry["phase"] != "after":
            continue
        with np.load(folder / entry["activity"]) as f:
            instant = f["instantaneous"]
            if entry["delay"] == 160 and entry["prefix"] == 4:
                states = f["mean"][-1].copy()
                resource_min = float(f["mean"][:,:,4].min())
                case_defs = read((folder/entry["activity"]).with_suffix(".json"))["cases"]
                frame = neurons[["bodyId","parameter_group","direct_input","observed_by_policy"]].copy()
                frame["M0_discovery_score"] = scores["scores"]
                for k, field in enumerate(("spikes","g","v","adapt","res")):
                    frame[field+"_mean"] = states[:,k].mean(0)
                for i,c in enumerate(case_defs):
                    if c["cue"] != 0:
                        continue
                    j=next(j for j,v in enumerate(case_defs) if v["cue"]==1 and v["context"]==c["context"] and v["visible"]==c["visible"])
                    delta=states[j,1:]-states[i,1:]
                    label=f"c{c['context']}-visible{int(c['visible'])}"
                    for k,field in enumerate(("g","v","adapt","res")):
                        frame[label+"-delta-"+field]=delta[k]
                    standardized=np.sqrt(np.mean((delta/scale)**2,axis=0))
                    frame[label+"-discovery-scaled-cue-distance"]=standardized
                    cell_rows.append(dict(model=name,context=c["context"],visible=c["visible"],cells=len(frame),
                        nonzero_voltage_cue=int((np.abs(delta[1])>1e-6).sum()),mean_abs_voltage_cue=float(np.abs(delta[1]).mean()),
                        nonzero_adaptation_cue=int((np.abs(delta[2])>1e-6).sum()),resource_min=resource_min,
                        nonzero_resource_cue=int((np.abs(delta[3])>1e-6).sum())))
                frame.to_parquet(O / (name+"-memory-cells-d160-p4.parquet"),index=False)
        trained = entry["delay"] in (0,40,160) and entry["prefix"] in (0,1,4)
        for case in instant:
            episodes.append((trained,((case-mean)/std)@basis))
        with np.load(folder/entry["readout"]) as a,np.load(folder/before[(entry["delay"],entry["prefix"])]["readout"]) as b:
            phase_feature_deltas.append(float(np.abs(a["features"]-b["features"]).max()))
    x=np.concatenate([z[:-1] for tr,z in episodes if tr]);y=np.concatenate([z[1:] for tr,z in episodes if tr])
    design=np.column_stack([x,np.ones(len(x))]);coef=np.linalg.solve(design.T@design+np.eye(design.shape[1]),design.T@y)
    for horizon in (1,20,40,80):
        pairs=[(z[:-horizon],z[horizon:]) for tr,z in episodes if not tr and len(z)>horizon]
        x=np.concatenate([a for a,b in pairs]);target=np.concatenate([b for a,b in pairs]);pred=x.copy()
        for _ in range(horizon):
            pred=pred@coef[:-1]+coef[-1]
        denominator=np.square(target-target.mean(0)).sum()
        fit_rows.append(dict(model=name,horizon_ms=horizon,
            affine_r2=float(1-np.square(target-pred).sum()/denominator),
            persistence_r2=float(1-np.square(target-x).sum()/denominator),
            histories=len(pairs),state_pairs=len(x),
            max_same_history_before_after_feature_delta=max(phase_feature_deltas),
            observation_only=True, fitted_on_training_histories=True))
pd.DataFrame(fit_rows).to_csv(O/"dynamics-persistence-check.csv",index=False)
pd.DataFrame(cell_rows).to_csv(O/"memory-cell-summary.csv",index=False)
with np.load(S/"runs/pathways/actual-direct-transmission.npz") as f:
    a=f["direct_current_into_observed"]
    for i,name in enumerate(f["group_labels"]):
        for cue in range(a.shape[1]):
            x=a[:,cue,i]
            flux_rows.append(dict(group=str(name),cue=cue,ms=len(x),signed_sum=float(x.sum()),abs_aggregate_sum=float(np.abs(x).sum()),
                peak_positive=float(x.max()),peak_negative=float(x.min()),nonzero_timepoints=int(np.count_nonzero(x))))
pd.DataFrame(flux_rows).to_csv(O/"observed-direct-flux.csv",index=False)
print(json.dumps({"dynamics_rows":len(fit_rows),"cell_summaries":len(cell_rows),"flux_rows":len(flux_rows)}))
