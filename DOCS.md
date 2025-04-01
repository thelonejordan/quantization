# Documentation

***The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin as compared to methods that seek to leverage human knowledge of the domain. One thing that should be learned from the bitter lesson is the great power of general purpose methods, of methods that continue to scale with increased computation even as the available computation becomes very great. The two methods that seem to scale arbitrarily in this way are search and learning.*** - [The Bitter Lesson (Sutton, 2019)](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)

In other words, the size of LLMs are going to grow inevitably as compute becomes increasingly accesible with time (Moore's Law).

## [`LLM.int8()`: 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)

### Computational Complexity of LLM Inference:

- Large pretrained language models with weights in multi-billions require significant memory for inference (directly proportional to floating point precision of weights)
- For large transformer language models at and beyond 6.7B parameters, the feed-forward and attention projection layers and their matrix multiplication operations are responsible for <u>95% of consumed parameters and 65-85% of all computation</u>. ([Ilharco et al., 2020](https://aclanthology.org/2020.emnlp-tutorials.4/))
- Other parameters come mostly from the embedding layer. A tiny amount comes from norms and biases.

### The Problem

One way to reduce the size of the parameters, thereby the computational complexity, is to quantize them to less bits and use low-bit-precision matrix multiplication. With this objective, 8-bit quantization methods for transformers have been developed:

- [A Statistical Framework for Low-bitwidth Training of Deep Neural Networks](https://arxiv.org/abs/2010.14298) (Chen et al., 2020)
- [Q8bert: Quantized 8bit bert](https://arxiv.org/abs/1910.06188) (Zafrir et al., 2019)
- [Q-bert: Hessian based ultra low precision quantization of bert](https://arxiv.org/abs/1909.05840) (Shen et al., 2020)
-  [Towards fully 8-bit integer inference for the transformer model](https://arxiv.org/abs/2009.08034) (Lin et al., 2020)

These studies investigate reducing the computational complexity of deep learning models in the training and inference space. While these methods reduce memory use, they degrade performance, usually require tuning quantization further after training, and have only been studied for models with less than 350M parameters.

***"Degradation-free quantization up to 350M parameters is poorly understood, and multi-billion parameter quantization remains an open challenge."***

Two key challenges:
1. the need for <u>higher quantization precision</u> at scales beyond 1B parameters
2. the need to explicitly represent the sparse but <u>systematic large
magnitude outlier features</u> that ruin quantization precision once they emerge in all transformer layers starting at scales of 6.7B parameters

<img style="margin: auto; display: block;" alt="Figure 1: OPT model mean zeroshot accuracy for WinoGrande, HellaSwag, PIQA, and LAMBADA datasets. Shown is the 16-bit baseline, the most precise previous 8-bit quantization method as a baseline, and our new 8-bit quantization method, LLM.int8(). We can see once systematic outliers occur at a scale of 6.7B parameters, regular quantization methods fail, while LLM.int8() maintains 16-bit accuracy." width=450 src="https://lh5.googleusercontent.com/TDrTlopijg4gvi2tTjPsMNLO23wcTJvhnYfc3WHszjtk5nsUgPQUxWOY5vyJysAomfhvbhgjhXP94sKT9v898vP53WW9ptb_itIpQ92xmkdfL7VHdY7cS1ldLpxh3parcz-lIdNgKL3NoxVXikqLfB0">

### The Proposed Solution

This paper shows empirically that, it is possible to perform inference in LLMs with up to 175B parameters without any performance degradation. For achieving this, the authors develop a procedure for Int8 matrix multiplication for feed-forward and attention projection layers in transformers, which cut the memory needed for inference by half while retaining full precision performance.

The first multi-billion-scale Int8 two-part quantization procedure for transformers that
does not incur any performance degradation:

1. **vector-wise quantization** with separate normalization constants for each inner product in the matrix multiplication, to quantize most of the features, that does not incur any performance degradation
2. **mixed-precision decomposition** scheme for the emergent outliers, which isolates the outlier feature dimensions into a 16-bit matrix multiplication while still more than 99.9% of values are multiplied in 8-bit

### Glossary

- low-bit-precision matrix multiplication
- quantization tuning
- Fully quantized training (FQT)
- quantization-aware training (QAT)

### Related papers
- [Integer quantization for deep learning inference: Principles and empirical evaluation](https://arxiv.org/abs/2004.09602) (Wu et al., 2020)
