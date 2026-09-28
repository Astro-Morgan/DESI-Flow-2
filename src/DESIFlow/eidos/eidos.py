"""
Eidos - The Autoencoder that hosts Plato (the hellinger metric encoder)
InceptionCNN + Transformer Masked Denoising Autoencoder
"""

# First order of business is architecture and training
# Is the latent representation staying a token sequence or pooled down to a legitimate low-d rep (changes Plato's input)
# does the decoder output an actual decoded spectrum or just the decoded masked tokens
# is it trained via MSE on decoded spectrum or decoded tokens or both
# Trained with small linear heads predicting certain observables to linearize the space wrt to those observables? (gradients orthogonalized against reconstruction so they can't interfere with information preservation)
# Plato trained at same time with orthogonalized gradients to same end?
# other metric encoders with orthogonalized gradients to promote diversity in the kernels and information content?