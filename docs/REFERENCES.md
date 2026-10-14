# References and method lineage

Bibliographic metadata was reviewed on 15 October 2026 against the supplied article, publisher pages, and author preprints. Issue years are used below; an online-first date can be earlier. Citation does not imply that every mechanism in a cited architecture is reproduced unchanged here. The [alignment map](PAPER_IMPLEMENTATION_MAP.md) describes the actual recovered implementation.

## Associated SERS paper

Sakib Mahmud, Faizul Rakib Sayem, Manal Hassan, Yanjun Yang, Muhammad E. H. Chowdhury, Susu M. Zughaier, Faycal Bensaali, and Yiping Zhao. **Deep learning-based cross-device standardization of surface-enhanced Raman spectroscopy for enhanced bacterial recognition.** *Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy* **347**, 126931 (2026). [DOI: 10.1016/j.saa.2025.126931](https://doi.org/10.1016/j.saa.2025.126931). Available online 19 September 2025. The author list and 2026 issue citation follow the supplied version of record; external indexes can differ.

## Operational neurons

1. Serkan Kiranyaz, Turker Ince, Alexandros Iosifidis, and Moncef Gabbouj. **Operational neural networks.** *Neural Computing and Applications* **32**, 6645–6668 (2020). [DOI: 10.1007/s00521-020-04780-3](https://doi.org/10.1007/s00521-020-04780-3). Generalized nodal/pooling operators; foundational context.
2. Serkan Kiranyaz, Junaid Malik, Habib Ben Abdallah, Turker Ince, Alexandros Iosifidis, and Moncef Gabbouj. **Self-organized Operational Neural Networks with Generative Neurons.** *Neural Networks* **140**, 294–308 (2021). [DOI: 10.1016/j.neunet.2021.02.028](https://doi.org/10.1016/j.neunet.2021.02.028); [author preprint](https://arxiv.org/abs/2004.11778). Learned polynomial nodal responses underlying the classifier's expansion.
3. Serkan Kiranyaz, Junaid Malik, Mehmet Yamac, Mert Duman, Ilke Adalioglu, Esin Guldogan, Turker Ince, and Moncef Gabbouj. **Super Neurons.** *IEEE Transactions on Emerging Topics in Computational Intelligence* **8**(1), 206–228 (2024). [DOI: 10.1109/TETCI.2023.3314658](https://doi.org/10.1109/TETCI.2023.3314658); [author preprint](https://arxiv.org/abs/2109.01594). Kernel-location shifts extend the generative-neuron receptive field. This repository preserves its supplied model's fractional sampling contract and does not silently substitute another SuperONN implementation.

## Reconstruction architecture context

4. Sakib Mahmud, Md Shafayet Hossain, Muhammad E. H. Chowdhury, and Mamun Bin Ibne Reaz. **MLMRS-Net: Electroencephalography (EEG) motion artifacts removal using a multi-layer multi-resolution spatially pooled 1D signal reconstruction network.** *Neural Computing and Applications* **35**, 8371–8388 (2023). [DOI: 10.1007/s00521-022-08111-6](https://doi.org/10.1007/s00521-022-08111-6). Prior 1D reconstruction architecture in the supplied translator's lineage; this is an EEG study, not SERS validation.
5. Huimin Huang, Lanfen Lin, Ruofeng Tong, Hongjie Hu, Qiaowei Zhang, Yutaro Iwamoto, Xianhua Han, Yen-Wei Chen, and Jian Wu. **UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation.** *ICASSP 2020*, 1055–1059 (2020). [DOI: 10.1109/ICASSP40776.2020.9053405](https://doi.org/10.1109/ICASSP40776.2020.9053405); [author preprint](https://arxiv.org/abs/2004.08790). Full-scale feature aggregation and deep-supervision context.
6. Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia. **Pyramid Scene Parsing Network.** *CVPR 2017*, 2881–2890 (2017). [DOI: 10.1109/CVPR.2017.660](https://doi.org/10.1109/CVPR.2017.660); [author preprint](https://arxiv.org/abs/1612.01105). Multi-scale spatial-pooling context.

## Explanation methods

7. Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. **Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps.** arXiv:1312.6034 (2013). [Author preprint](https://arxiv.org/abs/1312.6034). Input-gradient sensitivity maps adapted here to a fixed classifier target.
8. Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, and Martin Wattenberg. **SmoothGrad: removing noise by adding noise.** arXiv:1706.03825 (2017). [Author preprint](https://arxiv.org/abs/1706.03825). Averaging gradients over noise-perturbed inputs. This optional maintained explanation is not evidence of a recovered paper-trained checkpoint.

## Related spectral-model context

9. Nabil Ibtehaz et al. **RamanNet: a generalized neural network architecture for Raman spectrum analysis.** *Neural Computing and Applications* **35**, 18719–18735 (2023). [DOI: 10.1007/s00521-023-08700-z](https://doi.org/10.1007/s00521-023-08700-z). Related spectral-model work cited by the associated article; it is not a bundled baseline or an installed dependency.

The repository's software citation is [CITATION.cff](../CITATION.cff). Source-file credits, code licensing, and implementation boundaries are recorded in [NOTICE.md](../NOTICE.md).
