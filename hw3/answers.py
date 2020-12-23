r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=32,
        seq_len=64,
        h_dim=256,
        n_layers=3,
        dropout=0.4,
        learn_rate=1e-3,
        lr_sched_factor=0.5,
        lr_sched_patience=3,
    )
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = 'ACT I.'
    temperature = 0.0001  # 0.5
    # ========================
    return start_seq, temperature


part1_q1 = r"""
we split the corpus into sequences, in order to be able to iterate over the dataset and divide it into batches. 
"""

part1_q2 = r"""
the sequence length is only a technical separation, the actual "memory" of the model is the hidden state. 
since we only reset it at the end of the entire epoch, we can learn long sequences.
"""

part1_q3 = r"""
the model learns by predicting the next char in an auto-regressive way, so if we were to shuffle the data, the model would not learn to generate the continious text but would learn to generate noisy text that would make little sence.
we would shuffle the data if we would like to learn shortt sequences under the assumption that the sequences are independent, but then we would reset the hidden state at the end of each batch and not only at the end of the epoch.
"""

part1_q4 = r"""
1. we lower the temperature in order to reduce stochasticity and make the sampling more deterministic.
2. high temperature reduces the impact of the base distribution and pushes the distribution towards a uniform distribution.
3. low temperature increases the impact of the base distribution and pushes the distribution towards an argmax on the base distribution.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=16,
        h_dim=256,
        z_dim=4,
        x_sigma2=9e-3,
        learn_rate=2e-4,
        betas=(0.5, 0.99),
    )
    # ========================
    return hypers


part2_q1 = r"""


Sigma governs the relative strength of data-loss term with respect to KLD loss term. The larger the simga, the smaller the strengh of regression term, the larger the strength of KLD term and the level of variablity in generated images increases. This was evident in the following experiment in our model: Sigma^2 = 0.9: images shown little variace and tended towards an "average" image, all facing forward Sigma^2 = 0.0009 more weight to the data loss term: various faciel expressions, angles, face width
"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
