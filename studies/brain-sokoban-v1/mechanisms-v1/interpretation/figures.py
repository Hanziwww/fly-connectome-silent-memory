"""Render reviewed figure sets from saved summaries, with explicit denominators."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

O = Path(__file__).resolve().parent
S = O.parent
F = O / "figures"
F.mkdir(exist_ok=True)
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":140,
                     "axes.titlesize":11,"axes.labelsize":10,"legend.fontsize":8})
blue="#286f8f"; orange="#bf7844"; purple="#796091"; red="#ac4040"
colors=[blue,orange,"#4a9670",purple,red]
models=["remote-dynamic-M0-head_only","remote-dynamic-M0-joint","supp-M0-visibility","supp-S1-ring-brain","supp-S2-context-brain"]
labels=["M0 readout","M0 joint","M0 visible/hidden","S1","S2"]


def save(fig,name):
    for suffix in ("png","pdf"):
        fig.savefig(F/(name+"."+suffix),dpi=200,bbox_inches="tight")
    plt.close(fig)


d=pd.read_csv(S/"tables/run-index.csv");d=d[(d.module=="difficulty")&(d.origin=="local")]
fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
for tier,g in d.groupby(d.task.str[1].astype(int)):
    axes[0].scatter(np.full(len(g),tier)+np.linspace(-.1,.1,len(g)),g.training_seconds/60,color=blue,s=32)
axes[0].set(xlabel="Difficulty tier (3 maps each)",ylabel="Local training minutes",xticks=range(8),title="A  Acquisition on 24 maps")
c=pd.read_csv(S/"tables/training-controls.csv");c=c[(c.origin=="local")&c.task.isin(["D1-T0","D3-T0","D5-T0"])&c.condition.isin(["joint","brain_only","head_only","no_recovery","demonstration"])]
conds=["joint","brain_only","head_only","no_recovery","demonstration"]
for i,condition in enumerate(conds):
    g=c[c.condition==condition]
    for j,row in enumerate(g.itertuples()):
        axes[1].scatter(i+(j-1)*.13,row.training_seconds/60,marker="o" if row.original_solved else "x",color=colors[j],s=48)
axes[1].set(xticks=range(5),xticklabels=["Joint","Brain only","Readout only","No recovery","Full demo"],ylabel="Local minutes (log scale)",yscale="log",ylim=(.045,400),title="B  D1 / D3 / D5 controls")
axes[1].tick_params(axis="x",rotation=20)
axes[1].text(.02,.98,"Circle: solved; cross: stopped unsolved\nBlue / orange / green: D1 / D3 / D5",transform=axes[1].transAxes,va="top",fontsize=8)
save(fig,"01-acquisition")

b=pd.read_csv(O/"four-combination-behavior.csv",dtype={"combination":str});b=b[b.model.str.startswith("difficulty-")]
counts=b.groupby("combination").solved.sum().reindex(["00","10","01","11"])
dec=pd.read_csv(O/"decomposition-by-model.csv");dec=dec[dec.model.str.startswith("difficulty-")]
fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
axes[0].bar(range(4),counts,color=["#aeb9bf",orange,purple,blue]);axes[0].set(xticks=range(4),xticklabels=["Old B + old H","New B + old H","Old B + new H","New B + new H"],ylim=(0,27),ylabel="Maps solved / 24",title="A  Paired checkpoint combinations")
axes[0].tick_params(axis="x",rotation=15)
for i,v in enumerate(counts):axes[0].text(i,v+.4,f"{v}/24",ha="center")
for i,component in enumerate(["brain","head","interaction"]):
    values=dec[component].to_numpy();axes[1].scatter(i+np.linspace(-.16,.16,len(values)),values,color=[blue,orange,purple][i],s=20,alpha=.8)
    axes[1].plot([i-.22,i+.22],[np.mean(values)]*2,color="black",lw=2)
axes[1].axhline(0,color="#999",lw=.7);axes[1].set(xticks=range(3),xticklabels=["Brain","Readout","Interaction"],ylabel="Mean fixed-reference action margin change",title="B  Score decomposition: one dot per map")
save(fig,"02-brain-readout")

a=pd.read_csv(O/"neuron-weighted-atlas.csv");a=a[a.phase=="after"]
fig,axes=plt.subplots(1,3,figsize=(13,4),layout="constrained")
for i,pop in enumerate(["input","observed","other"]):
    g=a[a.population==pop]
    axes[0].scatter(i+np.linspace(-.12,.12,len(g)),g.active_cells/g.cells*100,color=colors[i],s=17,alpha=.7)
    axes[1].scatter(i+np.linspace(-.12,.12,len(g)),g.mean_abs_voltage_change,color=colors[i],s=17,alpha=.7)
for ax in axes[:2]:ax.set(xticks=range(3),xticklabels=["Input","Observed","Other"])
axes[0].set(ylabel="Cells with recorded spikes (%)",title="A  Sparse spiking in reference states")
axes[1].set(ylabel="Mean |voltage change| (mV)",title="B  Matched-state learning change")
cells=pd.read_csv(O/"memory-cell-summary.csv");cell=cells[(cells.model==models[0])&~cells.visible].iloc[0]
v=[cell.nonzero_voltage_cue,cell.nonzero_adaptation_cue,cell.nonzero_resource_cue]
axes[2].bar(range(3),np.asarray(v)/167106*100,color=[blue,orange,purple]);axes[2].set(xticks=range(3),xticklabels=["Voltage","Adaptation","Resources"],ylabel="Cells with cue difference > 1e-6 (%)",title="C  M0: delay 160 ms, prefix 4")
for i,x in enumerate(v):axes[2].text(i,x/167106*100+1,f"{int(x):,}",ha="center",fontsize=9)
axes[2].set_ylim(0,83)
save(fig,"03-neuron-atlas")

behavior=pd.read_csv(O/"memory-delay-summary.csv");geometry=pd.read_csv(S/"tables/memory-geometry.csv")
fig,axes=plt.subplots(1,3,figsize=(14,4),layout="constrained")
for m,label,color in zip(models,labels,colors):
    data=behavior[(behavior.model==m)&~behavior.visible];axes[0].plot(data.delay,data.solved/data.histories,"o-",color=color,label=label,ms=4)
    data=geometry[(geometry.model==m)&~geometry.visible&(geometry.phase=="after")].groupby("delay").readout_energy_fraction.mean()
    axes[1].plot(data.index,data*100,"o-",color=color,ms=4)
axes[0].set(xlabel="Delay (ms)",ylabel="Solved fraction across prefixes",ylim=(-.02,1.05),title="A  Complete hidden-goal behavior");axes[0].legend()
axes[1].set(xlabel="Delay (ms)",ylabel="Cue energy in readout row space (%)",title="B  Actual normalized policy features")
cf=pd.read_csv(O/"cross-family-memory-summary.csv")
for i,m in enumerate([models[0],models[3],models[4]]):
    g=cf[cf.evaluation_model==m]
    for held,offset,color in [(False,-.16,blue),(True,.16,orange)]:
        r=g[g.heldout_history==held].iloc[0];axes[2].bar(i+offset,r.correct/r.histories,.3,color=color,label=("Training timings" if not held else "Held-out timings") if i==0 else None)
axes[2].axhline(.5,ls="--",color="#888",lw=.8);axes[2].set(xticks=range(3),xticklabels=["M0","S1","S2"],ylim=(0,1.12),ylabel="Explanatory cue-decoder accuracy",title="C  Decoder fitted only on M0");axes[2].legend(loc="upper right")
save(fig,"04-memory-geometry")

c=pd.read_csv(O/"memory-intervention-summary.csv")
conditions=["read-voltage-clear","read-current-clear","read-adaptation-clear","read-resource-clear","read-delay-clear","read-all-clear","read-all-swap"]
cols=[(models[0],False),(models[2],False),(models[2],True),(models[3],False),(models[4],False)]
col_labels=["M0 hidden","M0 V/H hidden","M0 V/H visible","S1 hidden","S2 hidden"]
matrix=np.zeros((len(conditions),len(cols)));annotations=[]
for j,(m,visible) in enumerate(cols):
    for i,condition in enumerate(conditions):
        r=c[(c.model==m)&(c.visible==visible)&(c.condition==condition)].iloc[0]
        matrix[i,j]=r.solution_rate_change
        annotations.append((i,j,f"{r.net_change:+d}/{r.histories}" if isinstance(r.net_change,int) else f"{int(r.net_change):+d}/{int(r.histories)}"))
fig,axes=plt.subplots(1,2,figsize=(13,5),layout="constrained",gridspec_kw={"width_ratios":[1.5,1]})
im=axes[0].imshow(matrix,vmin=-1,vmax=1,cmap="RdBu",aspect="auto")
axes[0].set(yticks=range(7),yticklabels=["Clear voltage","Clear current","Clear adaptation","Reset resource","Clear delay buffer","Clear all","Swap all"],xticks=range(5),xticklabels=col_labels,title="A  Before first decision: paired solution change")
axes[0].tick_params(axis="x",rotation=25)
for i,j,label in annotations:axes[0].text(j,i,label,ha="center",va="center",fontsize=8,color="white" if abs(matrix[i,j])>.65 else "black")
fig.colorbar(im,ax=axes[0],shrink=.75,label="Fraction-point change")
for j,(m,visible) in enumerate(cols):
    for side,offset,color in [("indices",-.16,blue),("controls",.16,orange)]:
        condition=f"memory-g6546-{side}-1.0-hold";r=c[(c.model==m)&(c.visible==visible)&(c.condition==condition)].iloc[0]
        axes[1].bar(j+offset,r.solution_rate_change,.3,color=color,label=("IN17A052" if side=="indices" else "Matched cells") if j==0 else None)
axes[1].axhline(0,lw=.7,color="#888");axes[1].set(xticks=range(5),xticklabels=col_labels,ylabel="Paired solution-rate change",ylim=(-.48,.1),title="B  Full output suppression during delay/prefix");axes[1].tick_params(axis="x",rotation=25);axes[1].legend(loc="lower left")
save(fig,"05-causal-interventions")

paths=pd.read_csv(S/"tables/signed-pathways.csv");paths=paths[paths.line=="memory"]
fig,axes=plt.subplots(2,2,figsize=(13,8),layout="constrained")
names=["MNad18/27","SLP324","IN17A052"]
for a,col,title in [(axes[0,0],"direct_absolute","A  One-hop structural mass"),(axes[0,1],"two_hop_absolute","B  Two-hop structural mass")]:
    for i,group in enumerate(["memory-g8561","memory-g10181","memory-g6546"]):
        for side,offset,color in [("indices",-.17,blue),("controls",.17,orange)]:
            value=paths[(paths.group==group)&(paths.side==side)][col].iloc[0]
            a.bar(i+offset,value,.3,color=color,label=("Candidate" if side=="indices" else "Matched") if i==0 else None)
            if value==0:a.text(i+offset,1,"0",ha="center",fontsize=8)
    a.set(xticks=range(3),xticklabels=names,yscale="symlog",ylabel="Absolute path-weight mass",title=title);a.legend()
with np.load(S/"runs/pathways/actual-direct-transmission.npz") as f:
    times=f["milliseconds"];y=f["direct_current_into_observed"]
    for i in [4,5]:
        for cue in [0,1]:axes[1,0].plot(times,y[:,cue,i],color=blue if i==4 else orange,ls="-" if cue==0 else "--",alpha=.75,label=("Candidate" if i==4 else "Matched")+f", cue {cue}")
axes[1,0].set(xlabel="Simulation ms",ylabel="Signed direct current into observed pool",title="C  Actual M0 transmission, 160 ms / prefix 4");axes[1,0].legend(ncol=2)
dyn=pd.read_csv(O/"dynamics-persistence-check.csv")
for model,label,color in zip(models,labels,colors):
    d=dyn[dyn.model==model];axes[1,1].plot(d.horizon_ms,d.affine_r2,"o-",color=color,label=label,ms=4)
axes[1,1].axhline(0,lw=.8,color="#999");axes[1,1].set(xlabel="Forecast horizon (ms)",ylabel="Held-history R-squared",title="D  Affine dynamics: limited forecast horizon");axes[1,1].legend(fontsize=7)
save(fig,"06-pathways-dynamics")
print("Six reviewed figure sets rendered")
