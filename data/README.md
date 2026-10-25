# VitalDB Arrhythmia Database
**An Anesthesiologist-Validated Large-Scale Intraoperative Arrhythmia Dataset with Beat and Rhythm Labels**

## Abstract
Intraoperative cardiac arrhythmias present distinct characteristics and clinical challenges compared to non-surgical environments, yet publicly available electrocardiogram (ECG) databases have primarily focused on ambulatory and intensive care environments. To address this gap, we present the VitalDB Arrhythmia Database, a comprehensive collection of annotated intraoperative ECG recordings specifically designed for developing and validating arrhythmia detection algorithms in the surgical context.

The database comprises 734,528 seconds of continuous ECG data from 482 surgical patients, with over 660,000 individually annotated heartbeats classified across four beat types and 10 distinct rhythm categories. To efficiently process the extensive source data, we developed a custom deep learning beat classifier that served as an automated screening tool for arrhythmia candidate segments. All annotations underwent rigorous validation by five anesthesiologists, with each segment independently reviewed by at least two anesthesiologists. Inter-rater reliability analysis demonstrated excellent agreement with an overall Cohen's kappa of 0.930 ± 0.130.

This publicly accessible resource provides the research community with clinically validated intraoperative arrhythmia data, facilitating the development of robust detection algorithms suited to the unique physiological and technical challenges of the perioperative environment.

---

## Background
The intraoperative period is characterized by a high incidence of cardiac arrhythmias triggered by surgical stimuli, anesthetic agents, and autonomic fluctuations [1,2]. While intraoperative arrhythmias exhibit distinct patterns, existing public ECG datasets have predominantly consisted of data from ambulatory Holter recordings, failing to represent the unique characteristics of the surgical environment. The creation of such datasets has been challenging due to the need for experienced anesthesiologists to manually review hours of ECG recordings.

While most perioperative arrhythmias are benign, they can require immediate intervention, and delayed treatment can lead to increased morbidity and mortality. The scarcity of labeled arrhythmia datasets from intraoperative ECG data has hindered the development of sophisticated detection algorithms. To address this, we analyzed 734,528 seconds of ECG data from VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients [3,4], with rhythm and beat labels validated by anesthesiologists, and present the VitalDB Arrhythmia Database as a publicly accessible dataset to facilitate future medical research and algorithm development in this unique clinical context.

---

## Methods
The database was constructed through a multi-stage process involving automated screening of the public VitalDB open dataset [3,4], followed by meticulous manual annotation and validation by five anesthesiologists.

### 1. Automated Candidate Screening
A deep learning beat classifier, UniMS-ECGNet, was used to efficiently identify potential arrhythmia segments. The algorithm was trained on data from the VitalDB open dataset, the MIT-BIH Arrhythmia Database [3,5], and the CU Ventricular Tachyarrhythmia Database [3,6]. Candidate segments were selected based on criteria such as consecutive abnormal beats, high R-R interval variability, or patterned ectopy. Further details on the model architecture and usage can be found in the referenced repository [7].

### 2. Annotation and Validation
Individual ECG beats were classified into four primary classes (Normal, Supraventricular, Ventricular, Unclassifiable). Each continuous segment was assigned one of ten final rhythm labels. All automated selections underwent rigorous clinical validation by a board of five anesthesiologists, with each segment being independently labeled by at least two physicians to reach a final consensus.

---

## Data Description
The dataset includes a metadata file and 482 individual annotation files. The raw ECG waveform data is not included in this package but can be freely accessed and downloaded from the public VitalDB database using the corresponding `case_id` [3,4]. Alternatively, the waveform data can be accessed programmatically using the VitalDB Python package (`pip install vitaldb`) without requiring manual file downloads or additional authentication.

### Files
* **`metadata.csv`**: A single file summarizing all cases, including columns like `case_id`, `analyzed_duration_sec`, `total_beats`, and a list of `rhythm_classes` present in the case. Additionally, it includes relevant surgical and clinical information.
* **`Annotation_Files` Folder**: The folder contains a corresponding CSV file for each case, providing detailed beat and rhythm annotations. The annotation files are provided in the format `Annotation_file_[case_id].csv`.

### Column Definitions (Annotation Files)
* `time_second`: The timestamp of the R-peak in seconds, measured from the beginning of the recording.
* `beat_type`: The classification of the individual heartbeat.
* `rhythm_label`: The overall heart rhythm label for the segment in which the beat occurs.
* `bad_signal_quality`: A boolean marker (`True`/`False`) indicating if the beat is located within a segment of excessive noise or poor signal quality.
* `bad_signal_quality_label`: A label indicating the start or end of a bad signal quality segment (e.g., 'Start1', 'End1'). This column is empty for rows not marking these specific boundaries.

---

## Rhythm and Signal Quality Annotations

### Rhythm Classes
The database contains 10 distinct rhythm categories, with summary statistics presented below:

| Rhythm Label | Number of cases | Number of beats | Duration in Seconds |
|--------------|-----------------|-----------------|---------------------|
| **Normal Sinus Rhythm** | 370 | 408,420 | 384,407 |
| **Noise** | 250 | - | 67,734 |
| **Atrial Fibrillation** | 111 | 163,270 | 121,888 |
| **Patterned Ventricular Ectopy** | 109 | 24,069 | 47,481 |
| **Supraventricular Tachyarrhythmia** | 109 | 6,416 | 14,799 |
| **Ventricular Tachyarrhythmia** | 88 | 1,598 | 9,927 |
| **Patterned Atrial Ectopy** | 85 | 20,326 | 40,600 |
| **Sinus Node Dysfunction** | 66 | 23,141 | 31,942 |
| **Wandering Atrial Pacemaker / Multifocal Atrial Rhythm** | 26 | 10,132 | 9,630 |
| **Atrioventricular Block** | 10 | 4,323 | 5,486 |
| **Unclassifiable** | 6 | 199 | 631 |

### Signal Quality Labels
Specific criteria were used to label segments with poor signal quality:

* **Bad Signal Quality:** This flag was used for segments where QRS complexes are visible but noise obscures P-wave or T-wave morphology, making accurate interpretation difficult. This label can be used alongside other rhythm labels when the underlying rhythm remains interpretable.
* **Noise:** This label was applied to segments where artifacts are so severe that QRS complexes themselves cannot be detected. Therefore, segments labeled as 'Noise' do not contain beat-level annotations.

---

## Usage Notes: Accessing and Using the Data
The **VitalDB Arrhythmia Database** provides expert medical annotations for the ECG waveforms found in the comprehensive VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients, which is also available on PhysioNet [3, 4].

While our annotations are specific to ECG arrhythmias, users can easily link them to the full, raw waveform data from the original VitalDB project. This allows researchers to seamlessly integrate our labels with synchronized biosignals like PPG and arterial pressure waveforms, paving the way for novel multi-modal algorithm development.

A detailed, executable guide on how to merge and utilize both datasets is provided in the `UsageNote.ipynb` Jupyter Notebook in our [GitHub repository](https://github.com/vitaldb/arrdb/). The recommended workflow is as follows:

### 1. Download Annotations
Download the annotation files from this PhysioNet project.

### 2. Install Required Packages
Install the necessary Python packages to access waveform data and handle annotations.

```bash
pip install vitaldb pandas numpy matplotlib
```

### 3. Load ECG Waveforms and Annotations

Use the `case_id` to download the ECG waveform from VitalDB and load the corresponding annotation file.

```python
import vitaldb
import pandas as pd
import numpy as np

# Example case_id
case_id = 337

# Load ECG waveform from VitalDB (sampled at 500 Hz)
vals = vitaldb.load_case(case_id, ['SNUADC/ECG_II'], 1/500)
ecg_data = vals['SNUADC/ECG_II']

# Load annotation file
annotation_file = f'Annotation_file_{case_id}.csv'
annotations = pd.read_csv(annotation_file)

# Display the structure of annotations
print(annotations.head())

# Extract specific columns
time_seconds = annotations['time_second'].values
beat_types = annotations['beat_type'].values
rhythm_labels = annotations['rhythm_label'].values
signal_quality = annotations['bad_signal_quality'].values

# Example: Get annotations for a specific time range (e.g., 100-110 seconds)
start_time = 100
end_time = 110
segment_annotations = annotations[
    (annotations['time_second'] >= start_time) &
    (annotations['time_second'] <= end_time)
]

print(f"\nAnnotations between {start_time}s and {end_time}s:")
print(segment_annotations[['time_second', 'beat_type', 'rhythm_label']])

# Example: Filter by specific rhythm type
afib_beats = annotations[annotations['rhythm_label'] == 'Atrial fibrillation']
print(f"\nTotal Atrial fibrillation beats: {len(afib_beats)}")
```

### 4. Combine and Analyze

The annotation file contains beat-level information with timestamps (`time_second`), which can be matched with the ECG waveform data for visualization and analysis. Each `time_second` value corresponds to the R-peak location of a detected heartbeat.

---

## Limitations

* **Focused Annotation Scope:** The annotations in this dataset are concentrated on specific segments (approximately 20 minutes per case) identified as arrhythmia candidates by our screening process and finally labeled by anesthesiologists, rather than covering the full duration of anesthesia. Consequently, the dataset does not represent a continuous, exhaustive record of every beat throughout the entire surgical procedure.
* **Data Integration:** To facilitate efficient distribution, this package explicitly contains the expert annotations and metadata. It is designed to operate in tandem with the VitalDB [3,4], where the corresponding high-fidelity raw ECG waveforms are readily accessible for analysis.

---

## Release Notes

* **Version 1.0.0:** Initial release.

## Conflicts of Interest

The authors declare no competing interests.

## References

1. Kwon CH, Kim SH. Intraoperative management of critical arrhythmia. Korean J Anesthesiol. 2017;70(2):120-6.
2. Staikou C, Stamelos M, Stavroulakis E. Impact of anaesthetic drugs and adjuvants on ECG markers of torsadogenicity. Br J Anaesth. 2014;112(2):217-30.
3. Goldberger AL, Amaral LA, Glass L, Hausdorff JM, Ivanov PC, Mark RG, et al. PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals. Circulation. 2000;101(23):e215-20.
4. Lee HC, Park Y, Yoon SB, Yang SM, Park D, Jung CW. VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients. Sci Data. 2022;9(1):279.
5. Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng Med Biol Mag. 2001;20(3):45-50.
6. Nolle FM, Badura FK, Catlett JM, Bowser RW, Sketch MH. CREI-GARD, a new concept in computerized arrhythmia monitoring systems. Comput Cardiol. 1986;13:515-8.
7. VitalDB. Vital beat noise detection [Internet]. GitHub; 2025 [cited 2026 Jan 24]. Available from: https://github.com/vitaldb/arrdb/blob/main/Vital_beat_noise_detection.ipynb
8. VitalDB. arrdb. GitHub [Internet]. [cited 2026 Jan 24]. Available from: https://github.com/vitaldb/arrdb/