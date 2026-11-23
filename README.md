# Single-lead ECG analysis

`ecgProjectAnalysis.m` filters ECGs, trains a ForestFull rhythm classifier, and compares the waveforms and spectra of two reference excerpts and a project recording.

## Run in MATLAB

Use MATLAB R2025b with **Signal Processing Toolbox** and **Statistics and Machine Learning Toolbox**. Set MATLAB's Current Folder to the repository root, then run:

```matlab
ecgProjectAnalysis
```

To run individual sections in the Editor, start with Section 1. Run Section 2 before the BUT QDB prediction sections. All input data used by the script are included.

| Section | Analysis | Main outputs |
| --- | --- | --- |
| 1 | Filter all Challenge recordings with a 40 Hz zero-phase Butterworth low-pass and calculate 11 ECG features. | `filteredTraining`, `features` |
| 2 | Train a 150-tree ForestFull model on a stratified 80:20 recording split with seed 42. Evaluate it on the held-out recordings. | `forestFull`, `splitCounts`, `confusion`, `perClass`, `metrics`, `testPredictions` |
| 3–4 | Compare each BUT QDB excerpt before and after the 40 Hz low-pass. Plot waveforms and Welch PSDs, estimate heart rate, and report rhythm predictions. | `class1`, `class2`, `class1Predictions`, `class2Predictions` |
| 5 | Compare the project's raw ECG, inverted 40 Hz low-pass trace, and the low-pass trace followed by a 45–65 Hz band-stop. Plot waveforms and Welch PSDs and estimate heart rate. | `ownECG` |

## Data and interpretation

- **`training2017/`** contains the 8,528 public Challenge 2017 ECGs at 300 Hz with their WFDB headers. The script uses the bundled `REFERENCE.csv`, which has 5,050 normal, 738 AF, 2,456 other-rhythm, and 284 too-noisy labels. It reports accuracy, AF precision/recall/F1, Challenge F1 (mean of normal, AF, and other-rhythm F1), and macro F1 across all four classes. Data are from the [PhysioNet/Computing in Cardiology Challenge 2017](https://doi.org/10.13026/d3hm-sf11), licensed under [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/).
- **`butqdb/`** contains two unfiltered 30-second excerpts from BUT QDB record `100001` at 1,000 Hz. The MAT files retain the original ADC counts and include the calibration, class, and one-based start sample. Class 1 spans samples 128,674–158,673 and Class 2 spans 57,564–87,563, based on the source consensus annotations. Class numbers describe **signal quality**. The script resamples the excerpts to 300 Hz for ForestFull and reports an experimental filter-bypass prediction alongside the prediction using training preprocessing. Source: [Nemcova et al., BUT QDB v1.0.0](https://doi.org/10.13026/kah4-0w24), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See the [dataset paper](https://doi.org/10.1038/s41597-026-07905-w).
- **`recordings/ecg_20261123_151617_625.mat`** is the ADC recording used in Section 5. The script assumes a uniform **200 Hz** sample rate and **5 V** ADC reference. The voltage is the ADC input voltage. Heart-rate estimates depend on the assumed sampling rate and the detected QRS candidates.
