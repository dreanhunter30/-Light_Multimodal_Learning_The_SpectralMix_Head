# -Light_Multimodal_Learning_The_SpectralMix_Head
Catastrophic forgetting remains a major challenge in continual multimodal learning: when a model learns a new modality, it often overwrites previously acquired knowledge. In this report, we present SpectralMix, a lightweight classifier head with orthogonal components inspired by the analogy of light spectra. Just as ultraviolet, visible, and infrared bands coexist without destructive interference, spectral components allow the model to store modality-specific information separately.
On CIFAR-10, we combine frozen encoders (CLIP ViT-B/16 for images, DistilBERT for texts) with a SpectralMix head. Results show highly confident image classification (top-1 probability ≈ 0.997 for airplane), while text predictions remain less certain (top-1 ≈ 0.666). Analysis of mixing coefficients reveals that images activate a single dominant component, whereas texts distribute weights across several components. The Jensen–Shannon divergence between modalities is ≈ 0.63, confirming specialization.
This companion report illustrates the intuition behind SpectralMix and provides an accessible explanation complementing the formal preprint. Future work will explore larger datasets, stronger text encoders, and joint multimodal training.

Open demo: https://huggingface.co/spaces/Dreanhunter30/Light_Multimodal_Learning_The_SpectralMix_Head
Images: upload clear photos with a single object from the class list (default CIFAR-10: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
Image → upload photo → click Predict → check Top-5, charts, heatmap.
Text → enter a prompt ("a photo of a dog") → click Run.
Compare → upload photo + text → Compare to see alignment.  
Tip: keep the object large, clear, and without distracting background.(this demo upgrades a little bit, after this repot, metrics supposed to be good.)
