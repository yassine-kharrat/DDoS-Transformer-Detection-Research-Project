# Dataset: CIC-DDoS2019

## Overview

The [CIC-DDoS2019](https://www.unb.ca/cic/datasets/ddos-2019.html) dataset was created by the Canadian Institute for Cybersecurity. It contains network traffic captures of various DDoS attack types along with benign traffic.

## Attack Types (13 classes)

| Class | Description | Severity |
|---|---|---|
| BENIGN | Normal network traffic | — |
| DDoS-SYN | SYN flood attack | High |
| DDoS-UDP | UDP flood attack | High |
| DDoS-TCP | TCP flood attack | High |
| DDoS-HTTP | HTTP flood attack | Medium-High |
| DDoS-SlowLoris | Slowloris attack | Medium-High |
| DDoS-DNS | DNS amplification | Medium |
| DDoS-NTP | NTP amplification | Medium |
| DDoS-MSSQL | MSSQL amplification | Medium |
| DDoS-NetBIOS | NetBIOS attack | Medium |
| DDoS-LDAP | LDAP amplification | Medium |
| DDoS-SNMP | SNMP amplification | Medium |
| DDoS-SSDP | SSDP amplification | Medium |

## Features (43 selected)

After preprocessing (see below), 43 numeric network flow features are used, including:
- Packet size statistics (min, max, mean, std)
- Flow duration and inter-arrival times
- Protocol flags (SYN, ACK, RST, FIN counts)
- Byte and packet counts (forward/backward)
- Flow-level statistics

## Preprocessing Pipeline

### Step 1: Data Cleaning
- Remove rows with infinite or NaN values
- Drop constant or near-constant columns
- Remove the `SimillarHTTP`/`SimilarHTTP` column (inconsistent spelling, problematic values)
- Strip whitespace from column names

### Step 2: Data Balancing
The original dataset had severe class imbalance and was large enough to cause memory issues. A custom subsampling strategy was applied:

**Benign traffic:**
- Files with < 3,000 benign rows → kept all rows
- Files with ≥ 3,000 benign rows → capped at 3,000 rows per file
- Final benign set: **38,259 rows**

**Attack traffic:**
- 14 unique attack types selected (4 redundant DrDoS protocol-variant duplicates removed)
- Uniformly sampled **7,000 rows per attack type**
- Final attack set: **98,000 rows** (14 × 7,000)

**Final merged dataset:** 38,259 + 98,000 = **136,259 rows**, no duplicates
- Result: `balanced_cicddos2019.csv`

### Step 3: Train/Test Split
- Stratified split preserving class distribution
- Exported as separate files:
  - `preprocessed_train_X.csv` — Training features
  - `preprocessed_train_y.csv` — Training labels
  - `preprocessed_test_X.csv` — Test features
  - `preprocessed_test_y.csv` — Test labels

### Step 4: Sequence Creation (in code)
- Grouped flows by `Protocol` column
- Sorted by `Timestamp` within each protocol group
- Created sliding windows of 64 consecutive flows with 50% overlap (stride=32)
- Labels assigned via majority voting within each window
- Result: 3,052 training sequences, 760 test sequences

## File Placement

Place the preprocessed CSV files in this directory:

```
data/
├── README.md                          # This file
├── preprocessed_train_X.csv
├── preprocessed_train_y.csv
├── preprocessed_test_X.csv
└── preprocessed_test_y.csv
```

Or update the data paths in the notebooks to point to your file locations.

## Download

The raw dataset can be obtained from:
https://www.unb.ca/cic/datasets/ddos-2019.html

The preprocessing notebooks (`01_data_exploration.ipynb`, and the EDA notebooks in the original research directory) document the full cleaning pipeline. The balancing strategy is documented in `data_balancing.pptm`.
