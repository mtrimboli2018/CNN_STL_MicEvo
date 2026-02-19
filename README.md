## Introduction
This repository is the implementation of "Fully Convolutional Spatiotemporal Learning for Microstructure Evolution Prediction," which is a modified version of OpenSTLv0.3.0 developed by CAIRI AI Lab[^1]. This implementation focuses on the use of SimVPv2[^4] on microstructure evolution[^5] data.

## Getting Started
To create the enviornment, create an environment with "requirements.yaml." For exact repoducibility, "env-explicit.txt" may also be used to create the environment. Once the environment is created, activate the environment and run "python setup.py develop"

To run this network, use "CustomRuntine.py" with the provided data in this repository.

## Dataset
A sample of preprocessed data is the "data" folder of this repository. This is provided by Yang et. al. in "Self-supervised Learning and Prediction of Microstructure Evolution with Recurrent Neural Networks"[^5]. The full dataset can be accessed here:
https://data.mendeley.com/datasets/xdnjy9p5zn/1

## Acknowledgments
This repository is based on the work of CAIRI AI Lab[^1].

Original license: Apache 2.0 license

Original repository: https://github.com/chengtan9907/OpenSTL/tree/OpenSTLv0.3.0

Modifications include:
- Input sequence padding and output sequence trimming to fit the expected model input/output dimensions.
- Runtime on Python version 3.9.

## Citations

[^1]: C. Tan, S. Li, Z. Gao, W. Guan, Z. Wang, Z. Liu, L. Wu, S. Z. Li, Openstl: A comprehensive benchmark of spatio-temporal predictive learning, in: Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023.

[^2]: Z. Gao, C. Tan, L. Wu, S. Z. Li, Simvp: Simpler yet better video prediction, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 3170–3180.

[^3]: C. Tan, Z. Gao, S. Li, S. Z. Li, Simvp: Towards simple yet powerful spatiotemporal predictive learning, arXiv preprint arXiv:2211.12509 (2022).

[^4]: C. Tan, Z. Gao, L. Wu, Y. Xu, J. Xia, S. Li, S. Z. Li, Temporal attention unit: Towards efficient spatiotemporal predictive learning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 18770–18782.

[^5]: K. Yang, Y. Cao, Y. Zhang, M. Tang, F. Zhou, “RNN data”, Mendeley Data, V1, 2021, doi: 10.17632/xdnjy9p5zn.1

