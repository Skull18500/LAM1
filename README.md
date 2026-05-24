# TBAF
## Triangle Based Activation Function

| Original Image (1000x667) | TBAF Reconstruction (128x128 after 10k autoregressive image generations) |
| :--- | :--- |
| <img src="https://github.com/user-attachments/assets/0e663438-7f77-4120-a3f2-24afa1315e59" width="300"> | <img src="https://github.com/user-attachments/assets/fd77922a-ec06-4e29-9d5c-9119f34a63ea" width="300"> |

Standered Activation functions suffer from drift when used autoregressivly. **TBAF** *doesn't*. I have yet to find the mathematical limit, but i(and you, using the testLAM3.py) achieved 10k+ images generated autoregressivly, with the original image's key features completelly intact. I used the custom activation only once in the entire model, yet it completelly changed it the output's drift. In silu, past frame 500 the model began to break down, blurring features, and hallucinating. Past frame 10k with **TBAF**, you can see bright colored erroring at the edges of the screen, but that is due to CNN failures, not the activation function. The activation function is mathematically resistant to any changes, up to the precision of the format it is used in.
It can create an almost EXACT downscaled version of the original image after 10k images. It renders in 128 * 128 resolution. It, despite being trained on Dream's 4 hunters Finale manhunt, manages to reproduce almost any image perfectly.
The activation function is key. It creates a 'Triangle' out of 3 dimensions(out of the model's many) and uses the euclidian distances and such to remove mathematical error buildup, eliminating noise.
Another key fact, is this adds **0** parameters currently.

The **Triangle Based Activation Function (TBAF)** computes three pairwise Euclidean distances:

$$
\begin{aligned}
d_{g,12} &= \lVert \mathbf{v}_{g,1} - \mathbf{v}_{g,2} \rVert_2, \\
d_{g,13} &= \lVert \mathbf{v}_{g,1} - \mathbf{v}_{g,3} \rVert_2, \\
d_{g,23} &= \lVert \mathbf{v}_{g,2} - \mathbf{v}_{g,3} \rVert_2.
\end{aligned}
$$

The output block for group $g$ is a tensor of shape $(N, 3)$ where every row
contains the same three distances $[d_{g,12}, d_{g,13}, d_{g,23}]$.
Finally, all $G$ blocks are concatenated channel‑wise to obtain the output
$\mathbf{Y} \in \mathbb{R}^{N \times C}$:

$$
\mathbf{Y} = \bigoplus_{g=1}^{G}
\begin{bmatrix}
\lVert \mathbf{v}_{g,1} - \mathbf{v}_{g,2} \rVert_2 \\
\lVert \mathbf{v}_{g,1} - \mathbf{v}_{g,3} \rVert_2 \\
\lVert \mathbf{v}_{g,2} - \mathbf{v}_{g,3} \rVert_2
\end{bmatrix}_{(N,3)}.
$$


## Limitations:
I have only tested this in CNN based autoencoders, so i do not know how this could scale beyond that. Theoretically, we could achieve similar results in Vision-Transformers(ViTs), but that is untested.
As shown above, the corners of the image and the edges of UI objects tends to become very bright, but that is purely the CNNs failing to reconstruct the image properly.


One command and you can see the power unlocked by triangle distance manifolds:

`python testLAM3.py`
That will read a file named "download1.png", then generate 10k images autoregressivly saving to a video file, then it will create a "finalframe.png", the last frame(frame 10k).



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


If you can and are willing, please donate to help future projects like this. I am currently using an RTX 4060. Here is my KoFi page

https://ko-fi.com/skull18500
