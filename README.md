# Image Color Compression with K-Means Clustering
 
This project uses unsupervised learning to shrink a photo's color palette. Each pixel is treated as a point in 3D RGB space. K-means groups those points into *k* clusters, and every pixel is then repainted with its cluster's centroid color. As a result, a photo with roughly 190,000 distinct colors can be redrawn using only 10 or 50.
 
![Original image vs. k-means reconstructions at k = 1, 10, and 50](assets/comparison.jpg)
 
---
 
## How It Works
 
1. **Load the image.** The photo is 553 × 555 pixels with 3 RGB channels.
2. **Flatten it to a pixel table.** Reshape the array to `(306,915, 3)`, giving one row per pixel with `r`, `g` and `b` columns.
3. **Explore color space.** Plot every pixel as a point in an interactive 3D Plotly scatter, colored by its own RGB value.
4. **Cluster.** Fit `sklearn.cluster.KMeans` with `random_state = 67`.
5. **Reconstruct.** Replace each pixel with its cluster centroid, reshape back to the original dimensions and display the result.
6. **Inspect the palette.** Render each centroid as a color swatch, then re-plot the 3D scatter with each pixel colored by its assigned cluster.
The core of the method is the reusable `cluster_image(k)` function:
 
```python
def cluster_image(k, img=img):
    img_flat = img.reshape(img.shape[0] * img.shape[1], 3)
    kmeans = KMeans(n_clusters=k, random_state=67).fit(img_flat)
    new_img = img_flat.copy()
    for i in np.unique(kmeans.labels_):
        new_img[kmeans.labels_ == i, :] = kmeans.cluster_centers_[i]
    return new_img.reshape(img.shape)
```
 
---
 
## Results
 
| k | What you see |
|---|---|
| **1** | A single flat brown, the mean color of the whole image: RGB ≈ (105, 62, 53). |
| **10** | The scene is clearly recognizable but posterized. Flowers become flat patches of magenta, orange and crimson, and the blue thistles are lost. |
| **50** | Close to the original at a glance. Banding shows mainly in smooth gradients such as the rose petals. |
 
**The 10-color palette learned by k-means:**
 
![k = 10 cluster centroids](assets/palette_k10.png)
 
`rgb(48,47,38)` · `rgb(199,35,98)` · `rgb(225,142,13)` · `rgb(116,14,25)` · `rgb(142,133,132)` · `rgb(182,77,10)` · `rgb(62,84,106)` · `rgb(209,163,183)` · `rgb(108,88,56)` · `rgb(30,13,15)`
 
### Why this counts as compression
 
A raw RGB image stores 24 bits per pixel. After clustering, each pixel only needs an index into the palette:
 
| k | Bits per pixel | Approx. size (pixels + palette) | vs. raw (~921 KB) |
|---|---|---|---|
| 10 | 4 | ~153 KB | ~6× smaller |
| 50 | 6 | ~230 KB | ~4× smaller |
 
This is the same idea behind indexed-color formats like GIF and 8-bit PNG. The notebook demonstrates the concept and does not write compressed files.
 
---
 
## Repository Structure
 
```
.
├── k_means_compression.ipynb   # Full analysis
├── images.jpeg                 # Input image (floral arrangement)
├── images/
│   ├── comparison.jpg          # Original vs. k = 1 / 10 / 50 (README figure)
│   └── palette_k10.png         # k = 10 centroid swatches (README figure)
└── README.md
```
 
---
 
## Getting Started
 
**Requirements:** Python 3.9+
 
```bash
pip install numpy pandas matplotlib scikit-learn plotly jupyter
jupyter notebook k_means_compression.ipynb
```
 
The notebook reads `images.jpeg` from its own directory. To try a different photo, swap in any RGB JPEG or PNG and call `cluster_image(k)` with whatever *k* you like. Larger *k* values and larger images take noticeably longer to fit.
 
> **Note:** The 3D Plotly scatters draw all ~307K pixels, which makes them slow to render and very large when saved in the notebook. GitHub also won't display the interactive plots. To keep the repo light, clear outputs before committing (`jupyter nbconvert --clear-output --inplace k_means_compression.ipynb`) or sample the pixels before plotting (e.g. `img_flat_df.sample(20_000)`).
 

