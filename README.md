# Attention-Based DNN for Semantic-Importance-Aware Scheduling in IoT Networks

Machine Learning (25SC2107E) · Department of ECE, KL University · A.Y. 2026-27
Samanvitha Ramsetty (2520040024) · Hema Varshini (2520040068) · Ratna Manvitha (2520040007)

## What is in this folder

| Folder | Contents |
|---|---|
| `website/` | Flask website that runs the **trained models**: results dashboard, live wireless-network simulation, and a page that classifies uploaded TON_IoT records |
| `website/models/` | Trained models: `ab_dnn.keras`, `baseline_dnn.keras` (+ `.weights.h5` backups) and `preprocessing.joblib` (scaler, feature list) |
| `notebook/` | `ABDNN_TON_IoT_Wireless.ipynb` (run in Colab) and `ABDNN_TON_IoT_Wireless_executed.ipynb` (already run, all outputs visible) |
| `results/` | All charts (PNG) and result tables (CSV) for the slides |

## Run the website (Windows)

1. Install Python 3.10, 3.11 or 3.12 from python.org (tick **Add Python to PATH**).
2. Unzip this folder, open `website/`, and double-click **`run_windows.bat`**.
3. Wait for `Open http://127.0.0.1:5000`, then open that address in Chrome or Edge.

Mac / Linux: open a terminal in `website/` and run `sh run_mac_linux.sh`.

Manual way: `cd website` → `pip install -r requirements.txt` → `python app.py`.

The first install downloads TensorFlow (a few hundred MB). After that the site starts in about 10 seconds and works **offline**.

## What the website shows

1. **Top bar and hero**: project title, team, guide, and a diagram of the idea (IoT devices → AB-DNN → priority scheduler → base station).
2. **Headline banner**: urgent delay 13.07 → 1.22 ms, late urgent packets 56% → 0.34%, on-time data +33.1%.
3. **01 How it works**: the pipeline from TON_IoT record to network KPIs.
4. **02 Key results**: four improvement cards, the full network KPI table, the classification table, and the live re-check of the saved models' accuracy.
5. **03 Live network test**:
   - what FIFO / DNN / AB-DNN / Oracle mean
   - **Quick demo** buttons: Quiet (60%), Busy (90%), Overloaded (105%), Very bad signal (5 dB)
   - load and SNR sliders with a **Run simulation** button
   - four live graphs and a full table
   - an automatic explanation when the network is quiet or the signal is too weak
6. **04 Classify**: label 250 real TON_IoT records (or your own file) with the trained AB-DNN, with the top attended features.
7. **05 Charts** from the notebook, and **06 Method & dataset**.

No input is needed to start. The live test runs once automatically when the page opens.
Each live run is a single simulation, so numbers vary a little. For slides, quote the averaged results below
(load 0.9, SNR 20 dB, mean of 5 runs, from the notebook):
- Urgent (HIGH) delay 13.07 ms → 1.22 ms (**−90.7%**)
- Urgent packets late 56.0% → 0.34% (**−99.4%**)
- Useful on-time data 27.0 → 36.0 Mbps (**+33.1%**)

The notebook's full set of charts is in `results/` for your slides.

## Re-run the notebook in Colab

Upload `notebook/ABDNN_TON_IoT_Wireless.ipynb` to Colab → Runtime → Run all → choose `train_test_network.csv`
(TON_IoT, Train_Test_datasets → Train_Test_Network_dataset). Takes about 4–5 minutes; seed 42 reproduces these numbers.

## Honest notes for the viva

- Classification: AB-DNN 97.61% accuracy (macro F1 0.977), Baseline DNN 97.46% (F1 0.976). Over 3 seeds they are statistically tied.
  The network improvement comes from importance-aware scheduling; AB-DNN additionally shows **which traffic features** drive each decision.
- The IoT records are real (TON_IoT); the wireless channel and queue are simulated with standard models (QPSK, Rayleigh fading, HARQ).
- Importance levels come from the dataset's own `type` labels: normal → LOW; scanning, password, xss, mitm → MEDIUM; dos, ddos, injection, backdoor, ransomware → HIGH.

## Dataset citation

N. Moustafa, "A new distributed architecture for evaluating AI-based security systems at the edge: Network TON_IoT datasets," *Sustainable Cities and Society*, 2021, 102994.
A. Alsaedi, N. Moustafa et al., "TON_IoT telemetry dataset: a new generation dataset of IoT and IIoT for data-driven intrusion detection systems," *IEEE Access*, vol. 8, pp. 165130–165150, 2020.
