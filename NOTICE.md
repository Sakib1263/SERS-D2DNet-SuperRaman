# Research credits and implementation scope

This repository accompanies the SERS standardization paper by Sakib Mahmud, Faizul Rakib Sayem, Manal Hassan, Yanjun Yang, Muhammad E. H. Chowdhury, Susu M. Zughaier, Faycal Bensaali, and Yiping Zhao. DOI: [10.1016/j.saa.2025.126931](https://doi.org/10.1016/j.saa.2025.126931).

The translator's `translation/architecture.py` is a focused extraction of the author-supplied `MLMRSNet.IMUNet` graph, whose header names Sakib Mahmud and MIT licensing. Three inactive model builders were removed; active operations and weight shapes are retained and parity-tested against the original. The original full source SHA-256 is `9026c64849b1d27f171d69121684c00a3515eed2e8c175cca4d9d9bb94a19878`. The original copy remains in external recovery storage. Checkpoint metadata records its source hash and the separate focused-implementation hash.

The classifier is a new, focused implementation of the supplied model's tensor/state contract. It preserves power-major polynomial expansion, shared two-coordinate legacy sampling, spectral-axis concatenation, and layer state names. A separately named bounded 1D shift profile changes the legacy sampling behavior. It does not claim invention of the underlying operational neuron mechanisms or interchangeability with a newer SuperONN library.

The complete bibliography and each reference's role are listed in [docs/REFERENCES.md](docs/REFERENCES.md).

## Core operator references

- Kiranyaz et al., *Super Neurons*, IEEE Transactions on Emerging Topics in Computational Intelligence 8(1), 206-228 (2024), [10.1109/TETCI.2023.3314658](https://doi.org/10.1109/TETCI.2023.3314658).
- Kiranyaz et al., *Self-organized Operational Neural Networks with Generative Neurons*, Neural Networks 140, 294-308 (2021), [10.1016/j.neunet.2021.02.028](https://doi.org/10.1016/j.neunet.2021.02.028).
- Kiranyaz et al., *Operational neural networks*, Neural Computing and Applications 32, 6645-6668 (2020), [10.1007/s00521-020-04780-3](https://doi.org/10.1007/s00521-020-04780-3).

## Architecture and explanation context

- Mahmud et al., MLMRS-Net, [10.1007/s00521-022-08111-6](https://doi.org/10.1007/s00521-022-08111-6).
- Huang et al., UNet 3+, [10.1109/ICASSP40776.2020.9053405](https://doi.org/10.1109/ICASSP40776.2020.9053405).
- Zhao et al., Pyramid Scene Parsing Network, [10.1109/CVPR.2017.660](https://doi.org/10.1109/CVPR.2017.660).
- Simonyan, Vedaldi, and Zisserman, input-gradient saliency, [arXiv:1312.6034](https://arxiv.org/abs/1312.6034).
- Smilkov et al., SmoothGrad, [arXiv:1706.03825](https://arxiv.org/abs/1706.03825).

The release excludes the broad legacy classifier utility/baseline code and historical XAI tree. Installed dependencies retain their own licenses. The [dataset record](https://zenodo.org/records/23372983) separately lists CC BY 4.0 and restricts file access to approved requests. Research data and paper-trained weights are not bundled here. The article, dataset, software, and any separately supplied weights retain their respective license scopes. Synthetic fixtures are marked as such. Software correctness and benchmark measurements do not establish clinical efficacy or reproduction of published scores.
