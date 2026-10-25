# Search log — brain-pathology-v1 literature synthesis

Desk review of published, peer-reviewed findings on Drosophila models of age- and
disease-related decline in learning, memory and locomotion. Searches run 2026-10-19.

Sources used:
- PubMed via the bio-research MCP tools (`search_articles`, `get_article_metadata`,
  `get_full_text_article`, `convert_article_ids`).
- PubMed E-utilities (`esearch`/`esummary`/`efetch`) for complete title lists and abstracts
  when a result set was larger than the MCP tool returns in one call.
- PMC article pages (`https://pmc.ncbi.nlm.nih.gov/articles/<PMCID>/`) for open-access full
  text, converted to plain text locally.
- Europe PMC REST (`/search?query=EXT_ID:<pmid>`) to check full-text availability for one
  subscription-only paper (Riemensperger 2013).
- bioRxiv/medRxiv tools were available but not used: the bioRxiv API supports date- and
  category-browsing only (no keyword, author or full-text search), so it cannot be targeted at
  this topic, and the review was restricted to peer-reviewed primary papers in any case.

Inclusion rules applied: adult-fly studies with quantitative results; primary research only
(reviews used solely to locate primary papers); larva-only work excluded. Where a paper's
only measurements are in larvae, that is recorded in the atlas row and the mapping confidence
is marked low (e.g. 10.1093/braincomms/fcab049 larval NMJ active zones; the adult brain and
behaviour measurements from the same paper are the ones used).

## Primary topic searches

| ID | Query | Hits | Screened | Records included |
|----|-------|------|----------|------------------|
| Q1 | `drosophila AND (amyloid OR abeta OR Abeta42) AND (learning OR memory OR "mushroom body" OR locomotion OR excitability OR synapse OR "active zone")` | 306 | all 306 titles; 24 abstracts | 12 |
| Q2 | `drosophila AND tau AND (learning OR memory OR "mushroom body" OR locomotion OR excitability OR synapse OR "active zone")` | 178 | 60 titles after keyword filter; 15 abstracts | 9 |
| Q3 | `drosophila AND (aging OR ageing OR "age-related" OR aged) AND ("memory impairment" OR "age-dependent memory" OR "mushroom body" OR "active zone" OR excitability)` | 155 | 87 titles after filter | 9 |
| Q4 | `drosophila AND ("alpha-synuclein" OR synuclein OR PINK1 OR parkin) AND (learning OR memory OR "mushroom body" OR locomotion OR excitability OR synapse OR "active zone")` | 134 | 37 titles after filter; 13 abstracts | 6 |
| Q5 | `drosophila AND (dfmr1 OR fmr1 OR "fragile X") AND (learning OR memory OR "mushroom body" OR locomotion OR excitability OR synapse OR "active zone")` | 159 | 72 titles after filter | 4 |
| Q6 | `drosophila AND ("sleep deprivation" OR "sleep loss" OR "sleep restriction") AND (amyloid OR abeta OR tau OR synuclein OR PINK1 OR parkin OR fmr1 OR "fragile X" OR aging OR neurodegeneration) AND (learning OR memory OR locomotion OR synapse OR excitability)` | 22 | all 22 titles | 2 |

## Supplementary (physiology-targeted) searches

| ID | Query | Hits | Notes |
|----|-------|------|-------|
| S1 | `drosophila AND (aging OR aged OR "age-related" OR "age-dependent") AND ("giant fiber" OR "giant fibre" OR "neuromuscular junction" OR "flight motor" OR "escape response")` | 128 | found Martinez 2007 (giant fiber following frequency with age), Wagner 2015 (adult NMJ ultrastructure) |
| S2 | `drosophila AND (aging …) AND ("Kenyon cell" OR "mushroom body" OR "projection neuron") AND (electrophysiology OR "calcium imaging" OR "patch clamp" OR GCaMP OR excitability OR "synapse number" OR bruchpilot)` | 7 | found Cavaliere 2013 (KCNQ) |
| S3 | `drosophila AND (PINK1 OR parkin OR "alpha-synuclein" OR synuclein) AND (PPL1 OR PAM OR PPM1 OR "dopaminergic neuron…") AND (count OR number OR loss OR degeneration)` | 237 | 67 titles after filter; established the conflict between Pesah 2005 and section-based reports |
| S4 | `drosophila AND (amyloid OR abeta OR Abeta42) AND ("patch clamp" OR electrophysiology OR "calcium imaging" OR GCaMP OR "giant fiber" OR "release probability" OR bruchpilot OR "action potential" OR "potassium current")` | 19 | confirmed the Aβ electrophysiology set was complete (Ping 2015, Tabuchi 2015, Hahm 2018, Fang 2012, Lin 2014, Zhao 2010, Huang 2013) |
| S5 | `drosophila AND tau AND ("patch clamp" OR electrophysiology OR "calcium imaging" OR GCaMP OR "giant fiber" OR "release probability" OR bruchpilot OR "action potential" OR "electroretinogram")` | 20 | found Kadas 2019 (giant fiber system), Buhl 2019 (clock neuron firing), Chouhan 2016 (ERG) |
| S6 | `drosophila AND (dfmr1 OR fmr1 OR "fragile X") AND ("calcium imaging" OR GCaMP OR "patch clamp" OR electrophysiology OR "Kenyon cell" OR "giant fiber")` | 21 | found Franco 2017 (antennal lobe lateral inhibition), Gatto 2014, Tessier 2011 |

## Records included in the atlas (43 DOIs, 100 rows)

Starting set, all verified by PubMed metadata and included:

| DOI | PMID | Short reference | Full text read |
|-----|------|-----------------|----------------|
| 10.1371/journal.pgen.1005025 | 25774758 | Ping 2015, PLoS Genet | yes (PMC4361604) |
| 10.1016/j.neuroscience.2017.06.008 | 28627422 | Feng 2017, Neuroscience | abstract only (no PMC) |
| 10.1016/j.nbd.2012.11.006 | 23149068 | Huang 2013, Neurobiol Dis | abstract only (no PMC) |
| 10.1073/pnas.0400895101 | 15069204 | Iijima 2004, PNAS | yes (PMC article page) |
| 10.1101/lm.70804 | 15169857 | Mershin 2004, Learn Mem | yes (PMC419730) |
| 10.1523/JNEUROSCI.1374-22.2023 | 36868851 | Vourkou 2023, J Neurosci | yes (PMC10124960, incl. statistics table) |
| 10.3233/JAD-130617 | 23948901 | Beharry 2013, J Alzheimers Dis | abstract only (no PMC) |
| 10.1073/pnas.1118126109 | 22474396 | Tonoki & Davis 2012, PNAS | yes (PMC3341014) |
| 10.1523/JNEUROSCI.0978-14.2015 | 25609631 | Tonoki & Davis 2015, J Neurosci | yes (PMC4300323) |
| 10.1038/nn.3512 | 23995066 | Gupta 2013, Nat Neurosci | abstract only (no PMC) |
| 10.1371/journal.pbio.1002563 | 27684064 | Gupta 2016, PLoS Biol | yes (PMC5042543) |
| 10.3389/fnins.2010.00205 | 21165178 | White 2010, Front Neurosci | yes (PMC3002484) |
| 10.1093/sleep/32.8.984 | 19725249 | Seugnet 2009, Sleep | yes (PMC2717205) |
| 10.1038/s41598-025-34492-4 | 41501339 | Du 2026, Sci Rep | yes (PMC12864886) |

Added by this review:

| DOI | PMID | Short reference | Why included | Full text read |
|-----|------|-----------------|--------------|----------------|
| 10.1016/j.celrep.2018.06.029 | 29996096 | Hahm 2018, Cell Rep | sEPSC frequency in identified adult projection neurons, both directions with age | yes (PMC7505560) |
| 10.1016/j.cub.2015.01.016 | 25754641 | Tabuchi 2015, Curr Biol | sleep loss × Aβ interaction measured as l-LNv firing and K+ currents | yes (PMC4366315) |
| 10.1523/JNEUROSCI.3699-09.2010 | 20107079 | Zhao 2010, J Neurosci | synaptic fatigue and vesicle/mitochondrial counts in an identified adult neuron | yes (PMC6633795) |
| 10.1371/journal.pone.0177541 | 28520784 | López-Arias 2017, PLoS One | adult (not larval) active-zone counts for Aβ40/42/arc | yes (PMC5433741) |
| 10.1016/j.celrep.2018.11.098 | 30590040 | Coelho 2018, Cell Rep | adult-onset Aβ42 neuron loss with modifier tests | yes (PMC6315112) |
| 10.1091/mbc.E19-05-0303 | 31877058 | Arnés 2020, Mol Biol Cell | Aβ42 synapse loss with PI3K rescue | yes (PMC7183762, scanned) |
| 10.1371/journal.pone.0001703 | 18301778 | Iijima 2008, PLoS One | aggregation-dependent neuron vs neurite damage | abstract |
| 10.1186/s40478-016-0333-4 | 27338814 | Chouhan 2016, Acta Neuropathol Commun | standardised Aβ/tau/αSyn comparison in one adult cell type with percentages | yes (PMC4918017) |
| 10.3389/fncel.2019.00409 | 31551716 | Higham 2019a, Front Cell Neurosci | tau Ca²⁺ effects in MBONs and Kenyon cells with effect sizes | yes (PMC6746915) |
| 10.3389/fncel.2019.00260 | 31244615 | Higham 2019b, Front Cell Neurosci | MB excitability for Aβ42 and tau before degeneration | yes (PMC6581016) |
| 10.1523/JNEUROSCI.1490-09.2010 | 20071510 | Kosmidis 2010, J Neurosci | cell-type-specific tau toxicity in Kenyon cells | yes (PMC6632986) |
| 10.1016/j.nbd.2018.12.004 | 30529489 | Kadas 2019, Neurobiol Dis | isoform-specific presynaptic failure rate in the adult giant fiber system | abstract |
| 10.1016/j.nbd.2019.104507 | 31207389 | Buhl 2019, Neurobiol Dis | tau raises clock-neuron firing | abstract |
| 10.1007/s00401-013-1105-x | 23494099 | Wu 2013, Acta Neuropathol | tau in dopaminergic neurons, release loss before cell loss | abstract |
| 10.1371/journal.pone.0062445 | 23638087 | Cavaliere 2013, PLoS One | aging KCNQ/M-current loss with MB rescue | yes (PMC3640075) |
| 10.1038/s41467-019-09262-2 | 30899013 | Bhukel 2019, Nat Commun | MB autophagy block phenocopies aged active zones | yes (PMC6428838) |
| 10.1371/journal.pbio.3001730 | 36469518 | Sigrist lab 2022, PLoS Biol | "PreScale" trajectory across early aging | yes (PMC9721493) |
| 10.1016/j.redox.2024.103454 | 39642596 | 2025, Redox Biol | mitochondrial coupling to PreScale | abstract |
| 10.1002/cne.23798 | 25940748 | Wagner 2015, J Comp Neurol | independent adult synaptic ultrastructure with age | abstract |
| 10.3109/01677068409107077 | 6085635 | Technau 1984, J Neurogenet | historical MB fibre counts with age | abstract |
| 10.1002/dneu.20388 | 17443824 | Martinez 2007, Dev Neurobiol | age- and dfmr1-dependent giant fiber following frequency | abstract |
| 10.1016/j.celrep.2013.10.032 | 24239353 | Riemensperger 2013, Cell Rep | PAM-to-Kenyon-cell connection loss with α-synuclein | abstract only (not open access; checked via Europe PMC) |
| 10.1002/gene.20106 | 15789427 | Pesah 2005, Genesis | negative result: no dopaminergic cell loss | abstract |
| 10.1073/pnas.0501078102 | 15911761 | Whitworth 2005, PNAS | parkin dopaminergic degeneration with modifier | abstract |
| 10.1093/braincomms/fcab049 | 33997781 | 2021, Brain Commun | α-synuclein presynaptic changes before cell loss | yes (PMC8111063) |
| 10.1021/acschemneuro.2c00277 | 36576890 | 2023, ACS Chem Neurosci | parkin knockdown and dopamine release in aged flies | abstract |
| 10.7554/eLife.105386 | 42592915 | 2026, eLife | PINK1 loss and dopaminergic synapse loss | abstract |
| 10.1016/j.cub.2017.02.065 | 28366741 | Franco 2017, Curr Biol | dfmr1 weakens GABAergic lateral inhibition (patch clamp) | yes (PMC5405172) |
| 10.1016/j.nbd.2010.09.002 | 20843478 | Tessier 2011, Neurobiol Dis | dfmr1 mushroom body Ca²⁺ dynamics | yes (PMC2982942, scanned) |
| 10.1016/j.nbd.2014.01.008 | 24423648 | Gatto 2014, Neurobiol Dis | dfmr1 GABAergic circuit dysfunction | abstract |

## Excluded after screening (representative decisions)

- **Larva-only measurements**: 10.1242/dmm.012104 (Mhatre 2014, larval NMJ only);
  10.1002/neu.20227 and the companion modelling paper (Kidd 2006, cultured larval neurons with
  bath-applied peptide); 10.1093/hmg/ddr609 (parkin larval locomotion); 10.1038/ncomms15295 and
  10.1016/j.neuron.2018.01.022 (tau–synaptic-vesicle work, fly data from larval preparations).
- **Fly rather than human protein**: 10.1523/JNEUROSCI.1634-17.2017 (Drosophila Aβ/APPL and
  neprilysins), and the large APPL/APP-function literature.
- **Drug/intervention screens with no cell-type-resolved measurement**: the large set of
  plant-extract and small-molecule Aβ42 papers in Q1 (e.g. quercetin, curcuminoids, kaempferol).
- **Model not in scope**: APP:BACE models (10.1371/journal.pone.0020799, 10.1016/j.expneurol.2014.08.021)
  were screened and set aside because the state to be represented is the Aβ peptide itself; LRRK2,
  DJ-1, auxilin, VPS35 and GBA papers were set aside as outside the listed PD genes.
- **Reviews**: used only to locate primary papers (all rows in the atlas are primary research).

## Cell-type mapping reference

Mapping used `E:\my_projects\flybrain\manuscript\results\figure_data\fd_cells.csv` (read-only;
167,106 cells, 11,752 types). Counts that mattered for the mapping decisions:

- Kenyon cells: `cell_class == Kenyon_Cell`, 4,064 cells in 15 `KC*` types.
- `DPM` 2 cells, annotated **dopamine**; `APL` 2 cells, GABA.
- MBONs: 37 types, 97 cells (`MBON01`–`MBON35`); M4/M6 of the tau literature corresponds to the
  glutamatergic `MBON01`, `MBON03`, `MBON04`.
- Dopaminergic clusters: `PAM01`–`PAM15` (316 cells), `PPL101`–`PPL108`, `PPL201`–`PPL204`
  (`PPL203` is annotated serotonin), `PPM1201`–`PPM1205`, `PAL01`/`PAL03`.
- Antennal lobe: `cell_class == ALPN` 686 cells (projection neurons), `ALLN` 420 cells (local
  interneurons, GABAergic).
- Clock neurons: `l-LNv` 8 cells, annotated **serotonin**; `s-LNv` 8 cells, acetylcholine.
- Photoreceptors: `cell_class == visual`, 6,091 cells (`R1-R6`, `R7d`, `R7p`, `R7y`, …).
- Giant fiber pathway: `DNp01` (giant fiber), `PSI`, `TTMn`, `DLMn a, b`, `DLMn c-f`,
  `GFC1`–`GFC4`.
- Whole-model transmitter totals: dopamine 395 cells, serotonin 415, octopamine 141 — the three
  transmitter classes that carry zero weight in the model, which is what disqualifies several
  otherwise strong candidates.
