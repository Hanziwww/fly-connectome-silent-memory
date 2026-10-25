# Third-party code and data

| Component | Use | License | In this release |
|---|---|---|---|
| FlyBrain 0.1.0 (C. Benade; github.com/HEREISCB/flybrain, commit `{{FLYBRAIN_COMMIT}}`), with portions of flyverse (tel0s) | the whole-CNS spiking simulator; its model follows Shiu et al. (2024) | MIT | not included; cloned into `./flybrain` (see the README) |
| MaleCNS v1.0 connectome (Berg et al.; male-cns.janelia.org) | neuron annotations, transmitter predictions, connection weights, neuropil meshes and skeletons | CC BY 4.0 | not included in the code. The data deposit holds derived files (the simulator's connectome cache, the skeletons and the display geometry that the renders read), redistributed under CC BY 4.0 with this attribution |
| three.js and OrbitControls (the three.js authors) | the 3D view of the viewer (`sokoban/vendor/`) | MIT | included, with `sokoban/vendor/three-LICENSE.txt` |
| Single-cell atlases: Davie et al. 2018 (GEO GSE107451), Davis et al. 2020 (GSE116969), Özel et al. 2021 (GSE142787), Allen et al. 2020 (GSE141807; eLife 54074 source data), Lu et al. 2023 (GSE218661), Park et al. 2025 (GSE261656) | gene expression per cell type, joined to the model's cells | as published by their authors | not included; the deposit holds the per-unit summaries derived from them. `data/external/receipt.csv` lists the files used with their SHA-256 |

Cite the article, FlyBrain, the MaleCNS release and Shiu et al. (2024) when you use this code.
