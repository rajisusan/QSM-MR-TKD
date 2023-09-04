# QSM-MR-TKD
<p align="justify" markdown="1">
Raji Susan Mathew, Naveen Paluru and Phaneendra K. Yalavarthy, "Model-Resolution based Deconvolution for Improved Quantitative Susceptibility Mapping," NMR in Biomedicine (2023). (in press)

<p align="justify" markdown="1">
Quantitative susceptibility mapping (QSM) leverages the connection between the measured local field and the unknown susceptibility map to perform dipole deconvolution. A two-step approach is introduced that consists of first computing the TKD susceptibility map and then refining it using a model-resolution matrix. The TKD-derived susceptibility map is essentially a weighted average of the true susceptibility map, with the weights determined by the rows of the model-resolution matrix. Consequently, deconvolving the TKD susceptibility map with the model-resolution matrix provides a more precise approximation of the true susceptibility map. The model-resolution-based deconvolution employs closed-form, iterative, and sparsity-regularized methods.

#### Usage
```md
main.m
```

#### Sample data/metric computation credits
[1] Lai et al., Learned Proximal Networks for Quantitative Susceptibility Mapping, MICCAI, 2020. [https://github.com/Sulam-Group/LPCNN]

[2] Langkammer et al., Quantitative susceptibility mapping: report from the 2016 reconstruction challenge, Magnetic Resonance in Medicine, 2018. 

#### Any query, please raise an issue or contact :

Raji Susan Mathew

PostDoc, CDS, IISc Bangalore, email: rajisusanm@iisc.ac.in

Naveen Paluru

(PhD) CDS, MIG, IISc Bangalore, email: naveenp@iisc.ac.in

Prof. Phaneendra K. Yalavarthy

Professor, CDS, IISc Bangalore, email: yalavarthy@iisc.ac.in
