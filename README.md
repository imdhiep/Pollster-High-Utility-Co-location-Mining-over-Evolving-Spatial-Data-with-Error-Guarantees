# Pollster: High-Utility Co-location Mining over Evolving Spatial Data with Error Guarantees

Code and small test data for the paper *Pollster: High-Utility Co-location Mining over Evolving Spatial Data with Error Guarantees* (submitted to IEEE ICDE 2027).

Pollster estimates the utility participation index (UPI) of every co-location pattern from one utility-proportional threshold sample per feature. With probability at least 1 − δ, every estimate is within an error bound that grows with the square root of the UPI. Every instance has a persistent random seed. The sample at a snapshot therefore depends only on that snapshot, and rebuilding the sample or updating it incrementally gives the same output.

## Repository layout

| path | content |
|---|---|
| `src/pollster.py` | All experiments on TIST, Yelp, TIST-×r, and the FSQ-OS stream: exact miners (Star, Join, CQ), exact maintainers (Exact-Remine, Exact-Inc, Exact-Inc-LB, Exact-Count, EUCOLOC-U), and Pollster (rebuild and incremental modes). |
| `src/fsq_os_static.py` | Static FSQ-OS experiment (43.2 million places): exact mining over a permutation in chunks, and Pollster with threshold sampling. |
| `data/mini/`, `data/mini_dense/` | Small synthetic inputs in the TIST and Yelp file formats (20,000 POIs). `mini` is sparse; `mini_dense` places the POIs in an 8.4 km box (mean degree about 18). |
| `results/` | Summary of the churn experiment. |

The files in `src/` are byte-identical to the scripts that produced the reported runs:

| file | sha256 |
|---|---|
| `src/pollster.py` | `b01272eac9dc8df91c6af123f9ecbff37049dd975901764b68f9b678d0b4f3e5` |
| `src/fsq_os_static.py` | `000222e532279d3b5c850be03163847865b4e90f4a80c6cd81b726736368155e` |

## Requirements

Python 3.13, NumPy 2.1, Numba 0.61, SciPy, and pandas. The FSQ-OS experiments also need pyarrow and huggingface_hub.

```
pip install -r requirements.txt
```

The reported runs used Kaggle notebooks: 4 vCPUs (Intel Xeon, 2.2 GHz) and 30 GB of RAM.

## Quick start (synthetic data, under a minute)

```
# TIST-style probe: index, exact mining, Pollster, band verification, participation index
V9_ROOT=data/mini_dense/in V9_OUT=out V9_PARTS=probe python3 src/pollster.py

# FSQ-OS pipeline on 20,000 synthetic places
V18_FAKE=1 V18_MODE=exact    V18_KMAX=3 V9_OUT=out python3 src/fsq_os_static.py
V18_FAKE=1 V18_MODE=pollster V18_KMAX=3 V9_OUT=out python3 src/fsq_os_static.py
```

Results are written as JSON to `V9_OUT`, and progress lines are printed with the elapsed time.

## Experiments in `src/pollster.py`

The experiments are selected with `V9_PARTS`, a comma-separated list:

| part | experiment |
|---|---|
| `base`, `cqsmall` | Static exact miners on TIST shares, one thread; CQ at smaller distance thresholds |
| `util` | Utility distributions and sampling ablations (threshold, with replacement, uniform, self-normalized, priority) |
| `tile` | Scalability on TIST-×r, r = 1, 2, 4, 8 |
| `dvar` | Sensitivity to the distance threshold d (set `V9_K` for the number of features) |
| `mixed`, `ins`, `win` | TIST monthly streams: mixed insertions and deletions, insertion-only (with EUCOLOC-U), and a six-month window |
| `big` | Mixed stream on TIST-×8 |
| `yelp` | Yelp static run and quarterly stream |
| `probe` | Index build, Corollary 2 sample sizes, band verification, memory, participation index |
| `churn` | TIST streams with churn shares 1, 5, 20, and 100% |
| `topk`, `pioverlap` | Top-k tracking on the mixed stream; overlap of the PI and UPI samples |
| `fsqdelta` | FSQ-OS stream over three monthly releases |
| `fsqrep` | Repeated FSQ-OS static timings and the seed–utility correlation |

`progref`, `prtest`, and `fsqprog` (progressive refinement) are not used in the paper.

Other parameters: `V9_K` (number of features, default 60), `V9_D` (distance in meters, default 150), `V9_MS` (sample sizes), and `V9_IDLE` (months without a check-in before an instance is deleted, default 6).

## Data

- **TIST** (Yang, Zhang, and Qu, ACM TIST 2016): place `dataset_TIST2015_POIs.txt`, `dataset_TIST2015_Checkins.txt`, and `dataset_TIST2015_Cities.txt` under `V9_ROOT`. The files are found by name anywhere below this directory.
- **Yelp Open Dataset**: place `yelp_academic_dataset_business.json` and `yelp_academic_dataset_review.json` under `V9_ROOT`.
- **Foursquare Open Source Places** (`foursquare/fsq-os-places` on Hugging Face, Apache 2.0): the scripts download the monthly releases themselves. The dataset is gated, so a Hugging Face token with access is needed. The scripts read it from a Kaggle secret named `HF_TOKEN`. To run elsewhere, replace the `UserSecretsClient` call with your own way of supplying the token.

The files in `data/` are synthetic and only reproduce the input formats.

## Notes

- For the Kaggle runs, set `V9_ROOT` to the folder of the attached TIST and Yelp datasets (default `/kaggle/input`).
- Static seeds come from NumPy's PCG64 with seed 7. In the FSQ-OS stream, a place's seed is the top 53 bits of `SplitMix64(h ⊕ s)`, where `h` is a hash of its identifier, so the place keeps its seed across releases.
- Times exclude building the neighbor index, for every method.
