# RAMoE: Role-Aware Mixture-of-Experts for Short-Video Fake News Detection

Code for the paper **"Who is the Culprit? Role-Aware Mixture-of-Experts for Short-Video Fake News Detection"**.

## Training

Training is decoupled into two stages. Stage 1 trains the detector with the classification and unimodal losses only; the resulting **teacher** is used to
compute the role labels once; stage 2 trains the detector while the gate is supervised by those fixed labels.

**Stage 1 — teacher**

```
python main.py --dataset fakesv --epoches 10 --seed 2024 --use_attrib 0 
```

**Role labels**

```
python gen_labels.py --dataset fakesv 
```

**Stage 2 — attribution-supervised gate**

```
python main.py --dataset fakesv --epoches 20 --seed 2024 --use_attrib 1 
```

