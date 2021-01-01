import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        modules = []

        in_channels = self.in_size[0]
        out_channels = 1024

        modules.append(nn.Conv2d(in_channels, 128, kernel_size=4, stride=2, padding=1))
        modules.append(nn.BatchNorm2d(128))
        modules.append(nn.ReLU())
        # 128 x 32 x 32

        modules.append(nn.Dropout(0.2))
        modules.append(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1))
        modules.append(nn.BatchNorm2d(256))
        modules.append(nn.ReLU())
        # 256 x 16 x 16

        modules.append(nn.Dropout(0.2))
        modules.append(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1))
        modules.append(nn.BatchNorm2d(512))
        modules.append(nn.ReLU())
        # 512 x 8 x 8

        modules.append(nn.Dropout(0.2))
        modules.append(nn.Conv2d(512, out_channels, kernel_size=4, stride=2, padding=1))
        # out_channels x 4 x 4

        modules.append(nn.MaxPool2d(kernel_size=4))

        #FC layers
        self.cnn = nn.Sequential(*modules)
        self.linaer1 = nn.Linear(1024, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.linaer2 = nn.Linear(256, 1)


        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        x = self.cnn(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.linaer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        y = self.linaer2(x)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        modules = []

        self.in_channels = z_dim

        modules.append(nn.ConvTranspose2d(self.in_channels, 1024, kernel_size=4))#, stride=2, padding=1))
        modules.append(nn.BatchNorm2d(1024))
        modules.append(nn.ReLU())
        # 1024 x 4 x 4

        # modules.append(nn.Dropout(0.2))
        modules.append(nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2))#, padding=1))
        modules.append(nn.BatchNorm2d(512))
        modules.append(nn.ReLU())
        # 512 x 8 x 8

        # modules.append(nn.Dropout(0.2))
        modules.append(nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2))#, padding=1))
        modules.append(nn.BatchNorm2d(256))
        modules.append(nn.ReLU())
        # 256 x 16 x 16

        modules.append(nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2))#, padding=1))
        modules.append(nn.BatchNorm2d(128))
        modules.append(nn.ReLU())
        # 128 x 32 x 32

        # modules.append(nn.Dropout(0.2))
        modules.append(nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1))
        # 3 x 64 x 64

        # ========================
        self.cnn = nn.Sequential(*modules)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        if with_grad:
            z = torch.randn((n, self.z_dim), device=device, requires_grad=with_grad)
            samples = self.forward(z)
        else:
            with torch.no_grad():
                z = torch.randn((n, self.z_dim), device=device, requires_grad=with_grad)
                samples = self.forward(z)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        x = torch.tanh(self.cnn(z.view(z.shape[0], -1, 1, 1)))
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    device = y_generated.device
    norm_label_noise = 0.5 * label_noise

    min_value, max_value =  [data_label-norm_label_noise, data_label+norm_label_noise]
    diff = max_value-min_value
    noisy_data_label = min_value + torch.rand(y_data.shape, device=device)*diff

    min_value, max_value =  [1-data_label-norm_label_noise, 1-data_label+norm_label_noise]
    diff = max_value - min_value
    generated_label = min_value + torch.rand(y_generated.shape, device=device)*diff
    
    loss_func = torch.nn.BCEWithLogitsLoss(reduction='mean')
    
    loss_data = loss_func(y_data, noisy_data_label)
    loss_generated = loss_func(y_generated, generated_label)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    device = y_generated.device
    
    generated_labeles = torch.ones(y_generated.shape, device=device)
    generated_labeles = data_label * generated_labeles
    loss_func = torch.nn.BCEWithLogitsLoss()
    loss = loss_func(y_generated, generated_labeles) #compare generated probabilities to data labels - the objective is to fool the discriminator 
    # ========================
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: DataLoader,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    dsc_optimizer.zero_grad()
    
    #real images
    y_data = dsc_model.forward(x_data)
    
    #generate data w/o grad
    generated_data = gen_model.sample(x_data.shape[0], with_grad=False)

    #fake images w/o grad
    y_generated = dsc_model.forward(generated_data.detach())

    #fix y shape
    y_data = y_data.view(-1)
    y_generated = y_generated.view(-1)
    
    dsc_loss = dsc_loss_fn(y_data, y_generated)
    dsc_loss.backward(retain_graph=True)
    dsc_optimizer.step()

    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_optimizer.zero_grad()
    
    #generate data w/ grad
    generated_data_2 = gen_model.sample(x_data.shape[0], with_grad=True)

    #fake images w/ grad
    y_generated_2 = dsc_model(generated_data_2)

    #fix y shape
    y_generated = y_generated.view(-1)
    
    gen_loss = gen_loss_fn(y_generated_2.view(-1)) 
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    if len(gen_losses) < 11:
        return saved
    new_arr = []
    for i in range(len(gen_losses) - 10):
        new_arr.append(1/(abs(dsc_losses[i+10] - gen_losses[i+10])*(dsc_losses[i+10] + gen_losses[i+10])))
    tmp = max(new_arr)
    predicted_index = new_arr.index(tmp)
    if (predicted_index == len(gen_losses)-11):
        torch.save(gen_model, checkpoint_file)
        saved = True
    # ========================
    return saved
