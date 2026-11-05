# Near-Duplicate Document Detection (MinHash + LSH)

Find every pair of documents that are **at least 80% similar** in a collection of **112,571 documents**, in **under 15 minutes** on Google Colab's free CPU, using only **NumPy, SciPy, pandas and matplotlib**.

This repository contains the notebook `Task2_near_duplicates_v4.ipynb` for **Task 2: Near-Duplicate Detection** (20 marks, Colab, code marked).

---

## Table of contents
1. [What this project is (in simple words)](#1-what-this-project-is-in-simple-words)
2. [The task requirements](#2-the-task-requirements)
3. [How the method works](#3-how-the-method-works)
4. [The traps and how they are handled](#4-the-traps-and-how-they-are-handled)
5. [Repository contents](#5-repository-contents)
6. [Data files](#6-data-files)
7. [Setup and how to run (step by step)](#7-setup-and-how-to-run-step-by-step)
8. [What each notebook section does](#8-what-each-notebook-section-does)
9. [Outputs you get](#9-outputs-you-get)
10. [Using the function in your own code](#10-using-the-function-in-your-own-code)
11. [Parameters and why they were chosen](#11-parameters-and-why-they-were-chosen)
12. [Complexity of each stage](#12-complexity-of-each-stage)
13. [Test results](#13-test-results)
14. [Troubleshooting](#14-troubleshooting)
15. [Changes from v3 to v4](#15-changes-from-v3-to-v4)
16. [Glossary](#16-glossary)

---

## 1. What this project is (in simple words)

Imagine 112,571 news articles. Some are copies of each other, some are copies with a few words changed, and some just share the same header and footer. The goal is to list **every pair of articles that are almost the same**.

The obvious way, comparing every article with every other one, means **6.3 billion comparisons**. That takes more than a day, and the time limit is 15 minutes.

The notebook avoids that:

1. It gives every document a short **fingerprint** of 126 numbers.
2. It only compares documents whose fingerprints **partly match**, about 100,000 pairs instead of 6.3 billion.
3. It checks those few pairs **exactly**, so no wrong pair ever goes into the answer.

---

## 2. The task requirements

| Requirement | Value |
|---|---|
| Documents | 112,571 |
| Similarity measure | Jaccard similarity of word 5-gram shingles |
| Threshold | Jaccard ≥ 0.8 |
| Accuracy target | precision ≥ 0.9 and recall ≥ 0.9 |
| Time limit | under 15 minutes on Colab free CPU |
| Allowed libraries | NumPy, SciPy, pandas, matplotlib (plus the Python standard library) |

**Shingle rules (followed exactly):**
- Split text with `text.split()`, keeping case and punctuation exactly as they are.
- A shingle is 5 consecutive words.
- A document with fewer than 5 words gets **one** shingle made of all its words.

**Deliverables:**
- the notebook and `submission.csv`
- precision / recall / F1 on the dev set, plus 3 missed pairs and 3 wrong pairs with explanations
- a scaling experiment (n = 5k to 80k, log-log plot, slope from `np.polyfit`) and a brute-force comparison on small n
- the complexity of each stage
- parameter justification with the S-curve plot

---

## 3. How the method works

```
 documents ──► 1. shingles ──► 2. hash to integers ──► 3. MinHash signature (126 numbers per doc)
                                                                  │
 submission.csv ◄── 6. expand groups ◄── 5. exact Jaccard check ◄── 4. LSH banding (21 bands × 6 rows)
```

### Step 1: Shingling
`"the cat sat on the mat today"` becomes the 5-word shingles
`(the cat sat on the)`, `(cat sat on the mat)`, `(sat on the mat today)`.

Two documents are similar when they share most of their shingles:

```
Jaccard(A, B) = |shingles in both| / |shingles in either|
```

### Step 2: Hashing
Each shingle becomes one 64-bit integer, because numbers are much faster to compare than text. The hash is **deterministic**, so every run gives the same result. Documents with an identical set of shingles (exact duplicates) are grouped together here.

### Step 3: MinHash signatures
For each of 126 different hash functions, the smallest hash value over the document's shingles is kept. That gives a signature of **126 numbers per document**.

Key property: **the chance that two documents agree on a signature position equals their Jaccard similarity.** Two documents that are 90% similar agree on about 90% of the 126 numbers.

This step is fully **vectorised in NumPy**: each document's whole shingle array is processed against all 126 hash functions in one expression, with no Python loop over shingles.

### Step 4: LSH banding
The 126 numbers are split into **21 bands of 6 numbers**. For each band, documents with exactly the same 6 numbers fall into the same bucket. Any two documents that share a bucket in at least one band become **candidates**.

The probability that a pair with similarity `s` becomes a candidate is:

```
P(candidate) = 1 − (1 − s^6)^21
```

This is the **S-curve**. Pairs at 0.8 or above are caught 99.8%+ of the time, while unrelated pairs almost never are.

### Step 5: Exact verification
Every candidate is checked with the **exact Jaccard** on the full shingle sets. Only pairs with J ≥ 0.8 are kept. This makes **precision 1.0 by design**, because the only way to output a wrong pair would be a 64-bit hash collision.

Two cheap filters run first:
- **Size filter:** J can never be more than `smaller size / larger size`. This is exact, so it cannot lose a true pair.
- **Signature filter:** the pair is skipped if fewer than 60% of the 126 numbers agree. For a true pair (J ≥ 0.8) this is a 5.5-standard-deviation event, which essentially never happens.

### Step 6: Expand groups
Exact-duplicate groups found in step 2 are expanded back into all their pairs. A group of 3 identical documents gives 3 pairs.

---

## 4. The traps and how they are handled

| Trap | What could go wrong | How the notebook handles it |
|---|---|---|
| **Boilerplate** (thousands of articles share a header and footer) | Huge buckets, so m articles create m² pairs and the run explodes | (1) R = 6 rows per band: a shared header only puts two articles in one bucket if all 6 minimums come from the header (about 10⁻⁶ for 10% boilerplate). (2) Buckets bigger than 500 are not expanded in every band; their members go into one **boilerplate pool** that is screened **once** with a vectorised signature comparison. (3) The size and signature filters run before the exact check. (4) The signature is always built from the **full** shingle set, so the S-curve formula stays exactly true. |
| **Short documents** (under 5 words, or 6–9 words) | Ordinary 5-gram code breaks or gives no shingles | Under 5 words → one shingle of all the words, so two short docs are either identical (J = 1) or different (J = 0). 6–9-word docs get 2–5 shingles and go through the normal pipeline. |
| **Exact duplicates**, including groups of 3 | Missing some of the pairs in a group | Identical shingle sets are grouped first (compared exactly, not by a fingerprint), and every pair in the group is output. |
| **Pairs just below 0.8** (e.g. 0.79) | A fingerprint-only method would accept them | Every candidate is verified with the exact Jaccard. |
| **Speed** (about 100 million shingles) | Plain Python loops would take hours | Hashing, 5-gram combining and MinHash are vectorised in NumPy; band bucketing uses sorting. |

---

## 5. Repository contents

| File | What it is |
|---|---|
| `Task2_near_duplicates_v4.ipynb` | The complete notebook: code, experiments, plots and auto-generated report |
| `README.md` | This file |
| `submission.csv` | *(created when you run the notebook)* the answer: one row per similar pair, `id1,id2` with `id1 < id2` |

---

## 6. Data files

The data is **not stored in this repository** because `docs.csv` is about 480 MB, too big for GitHub. Cell 1 downloads it automatically from this Google Drive folder:

`https://drive.google.com/drive/folders/1VJycVGjec7N9rSRSdid8_wX36DtHgK5K`

| File | Size | Contents |
|---|---|---|
| `docs.csv` | ~480 MB | 112,571 documents, columns `id,text` |
| `dev_docs.csv` | ~21 MB | 5,059 practice documents, columns `id,text` |
| `dev_ground_truth.csv` | ~10 KB | the correct answer for the dev set: `id1,id2,jaccard` (341 pairs) |
| `ground_truth_pairs.csv` | ~175 KB | *(optional)* answer key for the full set; only used for a self-check, **never** by the algorithm |
| `stats.json` | tiny | counts of each document type the builder generated |

The folder must be shared as **"Anyone with the link → Viewer"** for the automatic download to work.

---

## 7. Setup and how to run (step by step)

No installation is needed on your computer. Everything runs in Google Colab.

### 7.1 Open the notebook in Colab
**Option A, from GitHub:**
1. Go to https://colab.research.google.com
2. Click **File → Open notebook → GitHub**.
3. Paste `QURSAM-70179225/NearDuplicateFileDetection` and press Enter.
4. Click `Task2_near_duplicates_v4.ipynb`.

**Option B, from your computer:**
1. Download `Task2_near_duplicates_v4.ipynb` from this repository.
2. In Colab: **File → Upload notebook** and choose the file.

### 7.2 Connect to a runtime
Click **Connect** (top right) and wait for the green tick. The default **CPU** runtime is what the task requires; you do not need a GPU.

### 7.3 Run Cell 1 (data)
Run the first code cell with **Shift + Enter**. It will:
- look for the three CSV files on the runtime,
- if missing, download the Google Drive folder into `/content/data` (using `gdown`, which Colab already has),
- load the files and print something like `docs=112571 dev=5059 dev_truth_pairs=341`.

The first download of the 480 MB file takes 1–3 minutes.

### 7.4 Run everything
Click **Runtime → Run all**.

| Part | Typical time on Colab free CPU |
|---|---|
| Section 1, full run + `submission.csv` | about 2–4 minutes (**this is the timed solution**) |
| Section 2, dev evaluation | under 30 seconds |
| Section 3, scaling experiment | about 5–8 minutes |
| Section 3, brute force (n up to 3,000) | about 2–3 minutes |
| Section 4, plots and ablation | about 1 minute |
| **Whole notebook** | **about 15–20 minutes** |

Only Section 1 counts toward the 15-minute limit. The other sections are extra experiments required for the report.

### 7.5 Download the results
1. Click the **folder icon** on the left side of Colab.
2. Right-click `submission.csv` → **Download**.
3. Save the notebook with outputs: **File → Download → Download .ipynb**, so the marker can see the plots and numbers.

---

## 8. What each notebook section does

| Section | Content |
|---|---|
| **Cell 1** | Imports, automatic data download, loading the CSV files |
| **0. Core implementation** | All pipeline stages as separate functions: `build_shingle_index`, `boilerplate_report`, `minhash_signatures`, `lsh_candidates`, `pool_candidates`, `verify_pairs`, `_find`, and the main function `find_similar_pairs` |
| **1. Full corpus** | Runs on all 112,571 docs, prints total time and per-stage times, writes `submission.csv`. If `ground_truth_pairs.csv` is present, also prints precision/recall on the full set. |
| **2. Dev report** | Precision, recall, F1 on the dev set; recall broken down by similarity range and document length; **3 missed + 3 wrong pairs with written explanations** (2a: errors of the real pipeline; 2b: errors of the same method without exact checking; 2c: closest calls); each example shows the first changed words |
| **3. Complexity & scaling** | Complexity table per stage; runtime at n = 5k, 10k, 20k, 40k, 80k (2 repeats each); brute force at n = 500–3,000; **log-log plot with slopes from `np.polyfit`**; stage breakdown; a written analysis filled in automatically with your measured numbers |
| **4. Parameters** | Table comparing band/row choices; **S-curve plot** (theory vs. measured on dev, normal and log scale); ablation table (bucket cap, filters, other R/B values, sketch-only); written justification filled in automatically |

All written reports are generated from the numbers measured during **your** run, so there are no blanks to fill in by hand.

---

## 9. Outputs you get

**`submission.csv`**
```
id1,id2
d000004,d002202
d000017,d000597
...
```
- one row per similar pair
- `id1 < id2` in every row (checked with an assertion)
- no duplicate rows (checked with an assertion)

**Printed in Section 1:**
```
TOTAL runtime: ...s   pairs found: ...
stage times (s): {'1 shingle+hash+dedupe': ..., '3 minhash': ..., '4 lsh': ..., '5 verify': ..., '6 expand': ...}
{'n_docs': 112571, 'n_reps': ..., 'n_shingles': ..., 'n_candidates': ..., 'oversize_buckets': ..., 'pool_size': ...}
boilerplate: {...}
```

**Plots:** runtime scaling (log-log) with fitted slopes and the 15-minute line; the S-curve in normal and log scale.

---

## 10. Using the function in your own code

After running Section 0 you can call the main function directly:

```python
ids   = ["a", "b", "c"]
texts = ["the quick brown fox jumps over the lazy dog",
         "the quick brown fox jumps over the lazy dog",
         "something completely different here today"]

pairs = find_similar_pairs(ids, texts, threshold=0.8)
print(pairs)              # {('a', 'b')}
print(LAST_DIAG["times"]) # time spent in each stage
```

To try different settings, use `_find` with `params`:

```python
pairs, diag = _find(ids, texts, threshold=0.8, params=dict(K=168, B=28, R=6))
```

---

## 11. Parameters and why they were chosen

Set at the top of Section 0 in `PARAMS`:

| Parameter | Value | Meaning | Why |
|---|---|---|---|
| `K` | 126 | numbers in each MinHash signature | MinHash cost grows with K; 126 keeps the full run to a few minutes |
| `B` | 21 | number of bands | `K = B × R` |
| `R` | 6 | rows per band | P(candidate) at J = 0.8 is **0.998**, while at J = 0.3 it is only **0.015**. Fewer rows (R = 4) would create about 15× more boilerplate candidates; more rows (R = 8) would drop recall at 0.8 to 0.947. |
| `MAX_BUCKET` | 500 | buckets bigger than this go to the boilerplate pool | far bigger than any natural duplicate group (a few docs), far smaller than a boilerplate group (thousands) |
| `EST_MIN` | 0.6 | minimum signature agreement before the exact check | 5.5 standard deviations below 0.8 for K = 126, so true pairs are never dropped |
| `SEED` | 12345 | random seed for the hash functions | reproducible results |
| `DF_FRAC`, `DF_MIN` | 0.002, 30 | a shingle in more than max(30, 0.2% of docs) is reported as boilerplate | used only for the report, not to change the signatures |
| `DOC_CHUNK` | 4000 | documents hashed together in one vectorised batch | limits memory use |

**Chosen band settings compared with the alternatives (all K ≈ 126):**

| B | R | P at J = 0.8 | P at J = 0.5 | P at J = 0.3 |
|---|---|---|---|---|
| 31 | 4 | 1.0000 | 0.865 | 0.223 |
| 25 | 5 | 1.0000 | 0.548 | 0.059 |
| **21** | **6** | **0.9983** | **0.282** | **0.015** |
| 18 | 7 | 0.9855 | 0.132 | 0.004 |
| 16 | 8 | 0.9470 | 0.061 | 0.001 |

---

## 12. Complexity of each stage

Notation: n = documents, L = average words per document, N ≈ n·L = total shingles, K = 126, B = 21, C = candidate pairs, m = boilerplate-pool size.

| Stage | Time | Memory |
|---|---|---|
| 1. Tokenise, hash, build 5-gram hashes | O(N) | O(N) |
| 1b. Unique shingles per doc + exact-duplicate grouping | O(N log L) | O(N) |
| 3. MinHash | **O(N·K)**, the bottleneck | O(n·K) |
| 4. LSH banding (sort per band) | O(n·K + B·n log n + C) | O(n + C) |
| 4b. Boilerplate pool (only if needed) | O(m²·K), done once | small, processed in blocks |
| 5. Verification | O(C·K + C′·L) | O(C) |
| 6. Expand groups | O(number of output pairs) | same |
| **Total** | **≈ O(N·K + N log L + B·n log n + C·(K + L))**, near-linear in n | |
| Brute force, for comparison | **Θ(n²·L)** | |

---

## 13. Test results

> **Important:** the numbers below come from a **synthetic test corpus** built to match the real data's statistics (`stats.json`): 112,571 documents, the same counts of each document type, shared headers and footers, short documents, and duplicate groups of 3. The real data could not be downloaded into the development environment. **Your real numbers appear in the notebook after you run it in Colab.**

| Measurement | Result (synthetic data, 2-core CPU) |
|---|---|
| Full run, 112,571 docs, ~66M shingles | **82 seconds** |
| Precision / recall / F1 on full set | **1.000 / 1.000 / 1.000** |
| Precision / recall / F1 on dev set | **1.000 / 1.000 / 1.000** |
| Candidate pairs checked | ~110,000 out of 6.3 billion possible (0.0017%) |
| Scaling slope (5k → 80k) | **1.04** (theory ≈ 1) |
| Brute-force slope | **1.98** (theory = 2) |
| Brute force extrapolated to 112,571 docs | about **31 hours** |
| Brute force vs. pipeline at n = 500–3,000 | **identical** pair sets |
| Bottleneck at n = 40,000 | MinHash ≈ 64%, shingling ≈ 35% |
| Sketch-only variant (no exact check), dev | precision 0.972, recall 0.959 |
| Extreme boilerplate stress test (3,000 tiny-body articles sharing a header) | recall 1.000 in 56 s |

---

## 14. Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: Still missing [...]` | Share the Drive folder as **Anyone with the link → Viewer** and run Cell 1 again. Or download the CSV files and drag them into the Colab file panel, then run Cell 1 again. |
| `gdown` says "Too many users have viewed or downloaded this file" | Google's download limit. Wait a while, or download the files manually and drag them into Colab. |
| `assert len(docs) > 100000` fails | Wrong or incomplete `docs.csv`. Re-download it, or re-run the dataset builder until it prints `[d] 112571 docs`. |
| Runtime disconnects or "RAM full" | Click **Runtime → Restart session** and run again. The full run needs about 4–5 GB of RAM; Colab free gives about 12 GB. |
| Run all is too slow | Run Cell 1, Section 0 and Section 1 only to get `submission.csv`. Then run the remaining sections. You can also reduce `BRUTE_SIZES` (e.g. `[500, 1000, 2000]`) or set `REPEATS = 1` in Section 3. |
| Different numbers from the table in section 13 | Expected. That table is from synthetic data; the notebook prints the real numbers. |
| Section 2 shows "0 missed, 0 wrong" | That is the expected result for the real pipeline. Sections 2b and 2c still give 3 + 3 explained examples, as the task requires. |

---

## 15. Changes from v3 to v4

| Problem in v3 | Fix in v4 |
|---|---|
| Used Python's `hash()`, which is salted per session, so the candidate count changed on every run (e.g. 79,617 then 149,714) | Deterministic hashing with `pandas.util.hash_array`; results are reproducible |
| Boilerplate was removed from some documents' signatures but not others (50% rule). Near-duplicate twins on opposite sides of the rule could not match, so 9 true pairs were missed on the synthetic set | Every document is signed with its full shingle set; boilerplate is handled with the bucket cap, the boilerplate pool and pre-filters. Recall 1.000. |
| Exact duplicates were grouped by a (size, sum, xor) fingerprint, so different documents could in theory be merged | Arrays are compared exactly |
| Written report had placeholders to fill in by hand | Reports are generated from the measured numbers |
| Fewer than 3 missed / wrong examples available | Sections 2b and 2c always give 3 + 3 explained examples |

---

## 16. Glossary

| Term | Meaning |
|---|---|
| **Shingle** | A group of 5 consecutive words from a document |
| **Jaccard similarity** | Shared shingles ÷ all distinct shingles of the two documents (0 = nothing in common, 1 = identical) |
| **Hash** | A function that turns text into a number |
| **MinHash** | A short signature of a document; two signatures agree in proportion to the Jaccard similarity |
| **LSH (Locality-Sensitive Hashing)** | Putting similar items into the same bucket so only those are compared |
| **Band / row** | The 126-number signature is cut into 21 bands of 6 rows each |
| **Candidate pair** | A pair that shares at least one bucket and is then checked exactly |
| **S-curve** | The graph of `1 − (1 − s⁶)²¹`: how likely a pair with similarity s is to be checked |
| **Boilerplate** | Text repeated across many documents, such as headers and footers |
| **Precision** | Of the pairs we output, the share that are correct |
| **Recall** | Of the true pairs, the share we found |
| **F1** | The balance (harmonic mean) of precision and recall |
| **Dev set** | A smaller practice dataset with a known answer, used to measure accuracy |
| **Brute force** | Comparing every document with every other one: correct but far too slow |

---

**Author:** Qursam Bibi, B.S. Information Engineering Technology, The University of Lahore
