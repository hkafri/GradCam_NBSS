> **Note:** This project focuses on XAI and builds upon the original NBSS repository by Audio-WestlakeU.

# Multichannel Speech Separation, Denoising and Dereverberation

The official repo of:  
[1] Changsheng Quan, Xiaofei Li. [Multi-channel Narrow-band Deep Speech Separation with Full-band Permutation Invariant Training](https://arxiv.org/abs/2110.05966). In ICASSP 2022.  
[2] Changsheng Quan, Xiaofei Li. [Multichannel Speech Separation with Narrow-band Conformer](https://arxiv.org/abs/2204.04464). In Interspeech 2022.  
[3] Changsheng Quan, Xiaofei Li. [NBC2: Multichannel Speech Separation with Revised Narrow-band Conformer](https://arxiv.org/abs/2212.02076). arXiv:2212.02076.  
[4] Changsheng Quan, Xiaofei Li. [SpatialNet: Extensively Learning Spatial Information for Multichannel Joint Speech Separation, Denoising and Dereverberation](https://arxiv.org/abs/2307.16516). TASLP, 2024.  
[5] Changsheng Quan, Xiaofei Li. [Multichannel Long-Term Streaming Neural Speech Enhancement for Static and Moving Speakers](https://arxiv.org/abs/2403.07675). IEEE Signal Precessing Letters, 2024.

Audio examples can be found at [https://audio.westlake.edu.cn/Research/nbss.htm](https://audio.westlake.edu.cn/Research/nbss.htm) and [https://audio.westlake.edu.cn/Research/SpatialNet.htm](https://audio.westlake.edu.cn/Research/SpatialNet.htm).
More information about our group can be found at [https://audio.westlake.edu.cn](https://audio.westlake.edu.cn/Publications.htm).

## Performance
SpatialNet: 
- Performance
  <br><img src="images/results.png" width="550">
- Computational cost
  <br><img src="images/model_size_and_flops.png" width="550">

## Requirements

```bash
pip install -r requirements.txt

# gpuRIR: check https://github.com/DavidDiazGuerra/gpuRIR
```

## Evaluation Example: Speaker Clustering & Reverberation

### Overview

This section demonstrates the **1D Time-Frequency Grad-CAM** interpretability method applied to the NBC2 (Narrow-band Conformer) speech separation model. The example shows how the network's attention patterns reveal speaker clustering and reverberation modeling without explicit positional encoding.

### Running the Grad-CAM Example

To reproduce the visualization shown below, run the test script on the included example audio:

```bash
python test_gradcam_nbc2.py \
  --audio-path examples/NBC2/0_mix_8_Full.wav \
  --output-figure gradcam_visualization.png \
  --output-heatmap gradcam_heatmaps.pt
```

### Visualization & Interpretation

![Grad-CAM 5-Panel Visualization](gradcam_nbc2_official_example_5panel.png)

#### Panel Structure

The above figure presents a 5-panel layout analyzing a highly reverberant audio mixture (RT60 = 0.8s) containing two speakers:

1. **Panel 1 (Top):** Input mixture spectrogram (magnitude in dB)  
   Shows the combined STFT of both speakers recorded in a reverberant environment.

2. **Panel 2 (Middle-Upper):** Separated STFT magnitude spectrogram for Speaker 1 (dB)  
   Represents the network's predicted speech for the first source.

3. **Panel 3 (Center):** Grad-CAM attention heatmap for Speaker 1  
   Visualizes which time-frequency regions were most important for separating Speaker 1. Warmer (yellow/red) regions indicate high attention, cooler (purple) regions indicate low attention.

4. **Panel 4 (Middle-Lower):** Separated STFT magnitude spectrogram for Speaker 2 (dB)  
   Represents the network's predicted speech for the second source.

5. **Panel 5 (Bottom):** Grad-CAM attention heatmap for Speaker 2  
   Shows the spatial attention pattern specific to Speaker 2 separation.

#### Key Findings

**Speaker Clustering:**
The Grad-CAM heatmaps (Panels 3 and 5) reveal **distinct and non-overlapping attention patterns** for each speaker. This demonstrates that the NBC2 model has learned to implicitly group time-frequency bins belonging to individual sources without explicit clustering labels. The network successfully discovers that Speaker 1's energy concentrates in certain frequency bands while Speaker 2 occupies different regions, a form of **learned source separation** through attention.

**Reverberation Modeling:**
In this highly reverberant scenario, both attention maps exhibit characteristic **"slanted lines" or "slash" patterns** rather than vertical or horizontal streaks. This observation indicates that:

- The deeper layers of NBC2 are **actively modeling the temporal smearing** introduced by room reverberation.
- The slanted patterns reflect the network's learned response to acoustic echoes and late reverberation tails, which spread energy across time at each frequency.
- Importantly, **no explicit positional encoding or hand-crafted temporal features** are used—the network learns reverberation dynamics purely from data through the Gram-CAM weighting of convolutional feature maps.
- The slants suggest the network is exploiting spectro-temporal dependencies, where speech of one speaker at time $t$ influences predictions at nearby time steps $t + \Delta t$, a key characteristic of reverberant audio.

This visualization validates that Conformer-based architectures like NBC2 can simultaneously achieve **speaker separation** and **reverberation awareness** through learned attention mechanisms, making them robust to real-world acoustic conditions.

## Generate Dataset SMS-WSJ-Plus

Generate rirs for the dataset `SMS-WSJ_plus` used in `SpatialNet` ablation experiment.

```bash
CUDA_VISIBLE_DEVICES=0 python generate_rirs.py --rir_dir ~/datasets/SMS_WSJ_Plus_rirs --save_to configs/datasets/sms_wsj_rir_cfg.npz
cp configs/datasets/sms_wsj_plus_diffuse.npz ~/datasets/SMS_WSJ_Plus_rirs/diffuse.npz # copy diffuse parameters
```

For SMS-WSJ, please see https://github.com/fgnt/sms_wsj

## Train & Test

This project is built on the `pytorch-lightning` package, in particular its [command line interface (CLI)](https://pytorch-lightning.readthedocs.io/en/latest/cli/lightning_cli_intermediate.html). Thus we recommond you to have some knowledge about the CLI in lightning. For Chinese user, you can learn CLI & lightning with this begining project [pytorch_lightning_template_for_beginners](https://github.com/Audio-WestlakeU/pytorch_lightning_template_for_beginners).

**Train** SpatialNet on the 0-th GPU with network config file `configs/SpatialNet.yaml` and dataset config file `configs/datasets/sms_wsj_plus.yaml` (replace the rir & clean speech dir before training).

```bash
python SharedTrainer.py fit \
 --config=configs/SpatialNet.yaml \ # network config
 --config=configs/datasets/sms_wsj_plus.yaml \ # dataset config
 --model.channels=[0,1,2,3,4,5] \ # the channels used
 --model.arch.dim_input=12 \ # input dim per T-F point, i.e. 2 * the number of channels
 --model.arch.dim_output=4 \ # output dim per T-F point, i.e. 2 * the number of sources
 --model.arch.num_freqs=129 \ # the number of frequencies, related to model.stft.n_fft
 --trainer.precision=bf16-mixed \ # mixed precision training, can also be 16-mixed or 32, where 32 can produce the best performance
 --model.compile=true \ # compile the network, requires torch>=2.0. the compiled model is trained much faster
 --data.batch_size=[2,4] \ # batch size for train and val
 --trainer.devices=0, \
 --trainer.max_epochs=100 # better performance may be obtained if more epochs are given
```

More gpus can be used by appending the gpu indexes to `trainer.devices`, e.g. `--trainer.devices=0,1,2,3,`.

**Resume** training from a checkpoint:

```bash
python SharedTrainer.py fit --config=logs/SpatialNet/version_x/config.yaml \
 --data.batch_size=[2,2] \
 --trainer.devices=0, \ 
 --ckpt_path=logs/SpatialNet/version_x/checkpoints/last.ckpt
```

where `version_x` should be replaced with the version you want to resume.

**Test** the model trained:

```bash
python SharedTrainer.py test --config=logs/SpatialNet/version_x/config.yaml \ 
 --ckpt_path=logs/SpatialNet/version_x/checkpoints/epochY_neg_si_sdrZ.ckpt \ 
 --trainer.devices=0,
```

## Module Version

| network | file |
|:---|:---|
| NB-BLSTM [1] / NBC [2] / NBC2 [3] | models/arch/NBSS.py |
| SpatialNet [4] | models/arch/SpatialNet.py |
| online SpatialNet [5] | models/arch/OnlineSpatialNet.py |

## Note
The dataset generation & training commands for the `NB-BLSTM`/`NBC`/`NBC2` are available in the `NBSS` branch.

