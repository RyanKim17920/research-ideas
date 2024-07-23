# Attention using KAN layers

KANs (Kolmogorov–Arnold Networks) were hailed as a possible replacement for linear/dense layers. While replacing MLP layers with KAN layers made the model worse, as tested in the [KAN-GPT](https://github.com/AdityaNG/kan-gpt) repository, I wanted to see if KAN layers could be used in attention mechanisms.
They ultimately did much worse in both efficiency and accuracy. 