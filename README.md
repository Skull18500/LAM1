# TBAF
## Triangle Based Activation Function
Standered Activation functions suffer from drift when used autoregressivly. **TBAF** *doesn't*. I have yet to find the mathematical limit, but i(and you, using the testLAM3.py) achieved 10k+ images generated autoregressivly, with the original image's ket featurescompletelly intact. I used the custom activation only once in the entire model, yet it completelly changed it the output's drift. In silu, past frame 500 the model began to break down, blurring features, and hallucinating. Past frame 10k with **TBAF**, you can see bright colored erroring at the edges of the screen, but that is due to CNN failures, not the activation function. The activation function is mathematically resistant to any changes, up to the precision of the format it is used in.
an almost EXACT downscaled version of the original image. It renders in 128 * 128 resolution. It, despite being trained on Dream's 4 hunters Finale manhunt, manages to reproduce almost any image perfectly.
The activation function is key. It creates a 'Triangle' out of 3 dimensions(out of the model's many) and uses the euclidian distances and such to remove mathematical error buildup, eliminating noise.
\section{Triangle-Based Activation Function (TBAF)}

We define the Triangle-Based Activation Function, $\text{TBAF}(\mathbf{x})$, as a geometric projection that maps latent channel triplets to their pairwise Euclidean distances. 

Let $\mathbf{x} \in \mathbb{R}^N$ be an input vector partitioned into $G = N/3$ groups, such that $\mathbf{x} = \{ \mathbf{g}_1, \dots, \mathbf{g}_G \}$ where each group $\mathbf{g}_i = [x_{i,1}, x_{i,2}, x_{i,3}]^T \in \mathbb{R}^3$. The output $\mathbf{y} = \text{TBAF}(\mathbf{x})$ is defined as the concatenation of pairwise distances within each group:

\begin{equation}
    \mathbf{y} = \bigoplus_{i=1}^{G} \begin{bmatrix} \|x_{i,1} - x_{i,2}\|_2 \\ \|x_{i,1} - x_{i,3}\|_2 \\ \|x_{i,2} - x_{i,3}\|_2 \end{bmatrix}
\end{equation}

where $\bigoplus$ denotes the concatenation operator across all groups $G$, and $\|\cdot\|_2$ represents the $L_2$ norm. This operation enforces translation invariance, as the output is determined by the relative distances between features rather than their absolute magnitudes, effectively acting as an attractor for the system's latent manifold.

| Original Image (1000x667) | TBAF Reconstruction (128x128 after 10k autoregressive image generations) |
| :--- | :--- |
| <img src="https://github.com/user-attachments/assets/0e663438-7f77-4120-a3f2-24afa1315e59" width="300"> | <img src="https://github.com/user-attachments/assets/fd77922a-ec06-4e29-9d5c-9119f34a63ea" width="300"> |



One command and you can see the power unlocked by triangle distance manifolds:

python testLAM3.py



##  Citation
If you use TBAF in your research or project, please cite it:

```bibtex
@misc{skull18500_tbaf_2026,
  author = {Skull18500},
  title = {Triangle-Based Activation Function (TBAF)},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/Skull18500/TBAF](https://github.com/Skull18500/TBAF)}}
}
