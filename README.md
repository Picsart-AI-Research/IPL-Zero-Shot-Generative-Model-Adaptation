# IPL-Zero-Shot-Generative-Model-Adaptation

This repository will be the official Pytorch implementation for [IPL](https://arxiv.org/abs/2304.).

**[Zero-shot Generative Model Adaptation via Image-specific Prompt Learning](https://arxiv.org/abs/2304.) (CVPR 2023)**
</br>
[Jiayi Guo](https://www.jiayiguo.net),
[Chaofei Wang](https://scholar.google.com/citations?user=-hwGMHcAAAAJ&hl=en&oi=ao),
You Wu,
Eric Zhang,
Kai Wang,
[Xingqian Xu](https://scholar.google.com/citations?user=s1X82zMAAAAJ&hl=en&oi=ao),
[Shiji Song](https://scholar.google.com/citations?user=rw6vWdcAAAAJ&hl=en&oi=ao),
[Humphrey Shi](https://www.humphreyshi.com),
[Gao Huang](https://www.gaohuang.net)
</br>
> **Abstract**: 
> Recently, CLIP-guided image synthesis has shown appealing performance on adapting a pre-trained source-domain generator to an unseen target domain. It does not require any target-domain samples but only the textual domain labels. The training is highly efficient, e.g., a few minutes. However, existing methods still have some limitations in the quality of generated images and may suffer from the mode collapse issue. 
A key reason is that a fixed adaptation direction is applied for all cross-domain image pairs, which leads to identical supervision signals. To address this issue, we propose an **I**mage-specific **P**rompt **L**earning (IPL) method, which learns specific prompt vectors for each source-domain image. This produces a more precise adaptation direction for every cross-domain image pair, endowing the target-domain generator with greatly enhanced flexibility. 
Qualitative and quantitative evaluations on various domains demonstrate that IPL effectively improves the quality and diversity of synthesized images and alleviates the mode collapse. Moreover, IPL is independent of the structure of the generative model, such as generative adversarial networks or diffusion models. 
<p align="center">
<img src="assets/fig2.png" width="500px"/></p>

## Overview

We propose a two-stage method named **I**mage-specific **P**rompt **L**earning (IPL) for generative model adaptation via only a text domain label. 
In Stage 1, a latent mapper is trained to produce a set of image-specific prompt vectors for each latent code of a source-domain image. In Stage 2, the trained latent mapper is plugged into the training process of the target-domain generator, and produces more precise and diversified adaptation directions for cross-domain image pairs. Our IPL largely addressed the mode collapse issue that appeared in existing works.
<p align="center">
<img src="assets/fig3.png" width="600px"/></p>


## Results

<p align="center">
<img src="assets/fig_res.png" width="600px"/></p>

## Citation

If you find our work helpful, please **star🌟** this repo and **cite📑** our paper. Thanks for your support!

```

```

## Contact
guo-jy20 at mails dot tsinghua dot edu dot cn


