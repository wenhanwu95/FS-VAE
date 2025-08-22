# Frequency-Semantic Enhanced Variational Autoencoder for Zero-Shot Skeleton-based Action Recognition [ICCV2025]
[Wenhan Wu](https://sites.google.com/view/wenhanwu/%E9%A6%96%E9%A1%B5), [Zhishuai Guo](https://zhishuaiguo.github.io/), [Chen Chen](https://www.crcv.ucf.edu/chenchen/), [Hongfei Xue](https://havocfixer.github.io/), [Aidong Lu ](https://webpages.charlotte.edu/alu1/)

[![arXiv](https://img.shields.io/badge/arXiv-2407.12322-00ff00.svg)](https://arxiv.org/abs/2506.22179)  [![Website](https://img.shields.io/badge/Website-Project%20Page-blue)](https://wenhanwu95.github.io/fsvae-project-page/)

[Video](https://www.youtube.com/watch?v=dN6b_EiHcCQ) [Poster](https://drive.google.com/file/d/1BJkDNc8kn1juclxAbWkoj0b0M6U8KqCL/view?usp=drive_link)

## Abstract
Zero-shot skeleton-based action recognition aims to develop models capable of identifying actions beyond the categories encountered during training. Previous approaches have primarily focused on aligning visual and semantic representations but often overlooked the importance of fine-grained action patterns in the semantic space (e.g., the hand movements in drinking water and brushing teeth). To address these limitations, we propose a Frequency-Semantic Enhanced Variational Autoencoder (FS-VAE) to explore the skeleton semantic representation learning with frequency decomposition. FS-VAE consists of three key components: 1) a frequency-based enhancement module with high- and low-frequency adjustments to enrich the skeletal semantics learning and improve the robustness of zero-shot action recognition; 2) a semantic-based action description with multilevel alignment to capture both local details and global correspondence, effectively bridging the semantic gap and compensating for the inherent loss of information in skeleton sequences; 3) a calibrated cross-alignment loss that enables valid skeleton-text pairs to counterbalance ambiguous ones, mitigating discrepancies and ambiguities in skeleton and text features, thereby ensuring robust alignment. Evaluations on the benchmarks demonstrate the effectiveness of our approach, validating that frequency-enhanced semantic features enable robust differentiation of visually and semantically similar action clusters, thereby improving zero-shot action recognition.

## Overall Design
![motivation](imgs/fig1.png)
The overall design of our frequency-semantic enhanced variational autoencoder for zero-shot skeleton action recognition. The main contributions are: 
- **Frequency Enhanced Module:**  
  We propose a Frequency Enhanced Module that employs Discrete Cosine Transform (DCT) to decompose skeleton motions into high- and low-frequency components, allowing adaptive feature enhancement to improve semantic representation learning in ZSSAR.

- **Semantic-based Action Description (SD):**  
  We introduce a novel Semantic-based Action Description (SD), comprising Local action Description (LD) and Global action Description (GD), to enrich the semantic information for improving the model performance.

- **Calibrated Cross-Alignment Loss:**  
  A Calibrated Cross-Alignment Loss is proposed to address modality gaps and skeleton ambiguities by dynamically balancing positive and negative pair contributions. This loss ensures robust alignment between semantic embeddings and skeleton features, improving the model's generalization to unseen actions in ZSSAR.

- **Extensive Experiments:**  
  Extensive experiments on benchmark datasets demonstrate that our framework significantly outperforms state-of-the-art methods, validating its effectiveness and robustness under various seen-unseen split settings.


## Our Approach
![Approach](imgs/fig2.png)
Overview of the proposed FS-VAE. The frequency-enhanced module integrates the global and fine-grained skeleton utilizing the low-frequency and high-frequency adjustments. The semantic-based action descriptions, including action labels, local action descriptions, and global action descriptions, are introduced to generate comprehensive semantic embeddings for cross-alignment. Moreover, the novel calibrated loss in the cross-alignment module is proposed for minimizing the disparity between semantic and skeletal features.

## Prerequisites:
- Environments: Please follow [**MSF-GZSSAR**](https://github.com/EHZ9NIWI7/MSF-GZSSAR/tree/master).
- Datasets: Please follow [**MSF-GZSSAR**](https://github.com/EHZ9NIWI7/MSF-GZSSAR/tree/master) to download and prepare for the skeleton features. The FS-VAE semantic features can be downloaded [here](https://drive.google.com/drive/folders/1dk0SiivOw3Hk8zhK4GgxjohMPWGsdPyY?usp=drive_link) and place it in the root directory after downloading.
- Training and evaluation: <code>bash fsvae_60.sh</code> for the training&testing on NTU-60; <code>bash fsvae_120.sh</code> for the training&testing on NTU-120

## Acknowledge
Mainly borrow from [**MSF-GZSSAR**](https://github.com/EHZ9NIWI7/MSF-GZSSAR/tree/master) and [**synse-zsl**](https://github.com/skelemoa/synse-zsl). Thanks for the great contributions!

## Citation
If you find this code useful for your research, please consider citing the following paper:

```bibtex

@article{wu2025frequency,
  title={Frequency-Semantic Enhanced Variational Autoencoder for Zero-Shot Skeleton-based Action Recognition},
  author={Wu, Wenhan and Guo, Zhishuai and Chen, Chen and Xue, Hongfei and Lu, Aidong},
  journal={arXiv preprint arXiv:2506.22179},
  year={2025}
}
```

For any questions, feel free to create a new issue or contact:
```
Wenhan Wu: wwu25@uncc.edu
```
