Ok, hear me out.:
Triangle based compression. We use triangles to render everything. But what happens if we use triangles for AIs? Well, as it turns out, i achieved 10k+ images generated autoregressivly, with the original image intact.
I used a custom triangle based activation function. Only once in the entire model, yet it completelly changed it. In silu, past frame 500 the model began to break down, blurring features. Past frame 10k, you can see 
an almost EXACT downscaled version of the original image.

One command and you can see the power unlocked by triangle distance manifolds:

python testLAM3.py
