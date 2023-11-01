# EdgeTran: Device-Aware Co-Search of Transformers for Efficient Inference on Mobile Edge Platforms

![Python Version](https://img.shields.io/badge/python-v3.6%20%7C%20v3.7%20%7C%20v3.8%20%7C%20v3.9-blue)
![Conda](https://img.shields.io/badge/conda%7Cconda--forge-v4.8.3-blue)
![PyTorch](https://img.shields.io/badge/pytorch-v1.8.1-e74a2b)

This repository constains the source code for the published IEEE Transactions on Mobile Computing paper. EdgeTran evaluates different transformer architectures on a diverse set of embedded platforms for various natural language processing tasks.
This repository uses the FlexiBERT framework ([jha-lab/txf_design-space](https://github.com/JHA-Lab/txf_design-space)) to obtain the design space of *flexible* and *heterogeneous* transformer models.

Supported platforms:
- Linux on x86 CPUs with CUDA GPUs (tested on AMD EPYC Rome CPU, Intel Core i7-8650U CPU and Nvidia A100 GPU).
- Apple M1 and M1-Pro SoC on iPad and MacBook Pro respectively.
- Broadcom BCM2711 SoC on Raspberry Pi 4 Model-B.
- Intel Neural Compute Stick v2.
- Nvidia Tegra X1 SoC on Nvidia Jetson Nano 2GB.

## Developer

[Shikhar Tuli](https://github.com/shikhartuli). For any questions, comments or suggestions, please reach me at [stuli@princeton.edu](mailto:stuli@princeton.edu).

## Cite this work

Cite our work using the following bitex entry:
```bibtex
@article{tuli2023edgetran,
  title={{EdgeTran}: Device-Aware Co-Search of Transformers for Efficient Inference on Mobile Edge Platforms},
  author={Tuli, Shikhar and Jha, Niraj K},
  journal={IEEE Transactions on Mobile Computing},
  year={2023}
}
```

## License

BSD-3-Clause. 
Copyright (c) 2022, Shikhar Tuli and Jha Lab.
All rights reserved.

See License file for more details.
