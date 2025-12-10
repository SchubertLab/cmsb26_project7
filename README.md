# Decoding Immune Memory: Predicting Disease Status from Single-Cell TCR Repertoires Using Machine Learning
*Supervisors: Benjamin Schubert (Lecturer)*

**Background**: The adaptive immune system, through T cell receptor (TCR) repertoires, encodes a
molecular “memory” of prior antigenic exposures (infections, vaccinations, autoimmunity, cancer) and
alterations in immune homeostasis [1,2,3]. Recent advances in single-cell technologies allow
simultaneous measurement of TCR sequences along with transcriptomes (scTCR + scRNA) for each
cell, enabling multimodal profiling of clonotype identity and functional state [4]. Meanwhile, machine
learning methods applied to TCR or BCR repertoires have shown promise in discriminating disease
conditions (e.g., COVID-19 severity, cancer vs healthy) from repertoire features [5,6,7,8]. Combining
single-cell TCR modalities with ML presents a compelling opportunity to exploit “immune-memory
imprint” as a diagnostic signature.

**Objective**: To develop and benchmark a machine learning pipeline that, given single-cell TCR data for
individuals, learns to classify disease status (e.g., autoimmune disease, latent viral reactivation, cancer)
by exploiting the imprint left in TCR clonotypes and associated cell states.

**Data**: The TCR-disease Atlas by Xiu et al. [3]: https://huarc.net/v2/download/ or simulated TCR
repertoire data with [9].

**Methodology**: Scanpy, Scirpy, scGEX and scTCR encoders, scikit-learn, PyTorch



## Literature: 
1. O’Donnell, T. J., Kanduri, C., Isacchini, G., Limenitakis, J. P., Brachman, R. A., Alvarez, R. A., ... & Greiff, V. (2024). Reading the repertoire: Progress in adaptive immune receptor analysis using machine learning. Cell Systems, 15(12), 1168-1189.
2. Katayama, Y., Yokota, R., Akiyama, T., & Kobayashi, T. J. (2022). Machine learning approaches to TCR repertoire analysis. Frontiers in immunology, 13, 858057.
3. Xue, Z., Wu, L., Gao, B., Tian, R., Chen, Y., Qi, Y., ... & Liu, W. (2025). A pan-disease and population-level single-cell TCRαβ repertoire reference. Cell Discovery, 11(1), 82.
4. Drost, F., An, Y., Bonafonte-Pardàs, I., Dratva, L. M., Lindeboom, R. G., Haniffa, M., ... & Schubert, B. (2024). Multi-modal generative modeling for joint analysis of single-cell T cell receptor and gene expression data. Nature Communications, 15(1), 5577.
5. Park, J. J., Lee, K. A. V., Lam, S. Z., Moon, K. S., Fang, Z., & Chen, S. (2023). Machine learning identifies T cell receptor repertoire signatures associated with COVID-19 severity. Communications Biology, 6(1), 76.
6. Weinstein, E. N., Wood, E. B., & Blei, D. M. (2024). Estimating the causal effects of T cell receptors. arXiv preprint arXiv:2410.14127.
7. Zaslavsky, M. E., Craig, E., Michuda, J. K., Sehgal, N., Ram-Mohan, N., Lee, J. Y., ... & Boyd, S. D. (2025). Disease diagnostics using machine learning of B cell and T cell receptor sequences. Science, 387(6736), eadp2407.
8. Slabodkin, A., Sollid, L. M., Sandve, G. K., Robert, P. A., & Greiff, V. (2023). Weakly supervised identification and generation of adaptive immune receptor sequences associated with immune disease status. bioRxiv, 2023-09.
9. Chernigovskaya, M., Pavlović, M., Kanduri, C., Gielis, S., Robert, P. A., Scheffer, L., ... & Greiff, V. (2025). Simulation of adaptive immune receptors and repertoires with complex immune information to guide the development and benchmarking of AIRR machine learning. Nucleic Acids Research, 53(3), gkaf025.
