# speech-separation
Target Source Separation

This repository contains the implementation of the SpEx+ model for the Target Source Separation task. 

# Results
SI-SDR = $9.05$ and PESQ = $1.68$ were achieved on public test split by the final model (using loudness normalization to -20).

# Installation
- Clone this repository
```bash
git clone https://github.com/hzchet/speech-separation.git
cd speech-separation
```
- Change `WANDB_API_KEY`, `SAVE_DIR`, `DATA_DIR` variables in the Makefile, to specify where you want to save logs/weights and also where you want to store your data.
- Build and run Docker container in the interactive mode by running the following command:
```bash
make build && make run
```
- Install the weights of the pre-trained model by running
```bash
python3 install_weights.py
```
- Copy the config into the same directory
```bash
cp configs/spex_plus.json saved/models/final/config.json
```

# Test
In order to run inference and measure PESQ and SI-SDR metrics, run
```bash
python3 test.py -r saved/models/final/config.json -t <test-folder-data> -b 1
```
