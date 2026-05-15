import torch
import torch.nn as nn
import torch.autograd as autograd

class Generator1D(nn.Module):
    def __init__(self, latent_dim, seq_len, channels):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        
        self.init_size = seq_len // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size)
        img = self.conv_blocks(out)
        return img

class Critic1D(nn.Module):
    def __init__(self, seq_len, channels):
        super().__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv1d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.25)]
            if bn:
                block.append(nn.BatchNorm1d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        ds_size = seq_len // 16
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size, 1))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

class BatteryWGANGP:
    def __init__(self, seq_len=512, channels=3, latent_dim=100, lr=0.0002, b1=0.5, b2=0.999):
        self.latent_dim = latent_dim
        self.generator = Generator1D(latent_dim, seq_len, channels)
        self.critic = Critic1D(seq_len, channels)
        
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        self.optimizer_C = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=(b1, b2))
        self.lambda_gp = 10

    def compute_gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1, device=real_samples.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.critic(interpolates)
        fake = torch.ones(real_samples.size(0), 1, device=real_samples.device, requires_grad=False)
        
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_step(self, real_data):
        batch_size = real_data.size(0)
        
        # Train Critic
        self.optimizer_C.zero_grad()
        z = torch.randn(batch_size, self.latent_dim, device=real_data.device)
        fake_data = self.generator(z)
        
        real_validity = self.critic(real_data)
        fake_validity = self.critic(fake_data)
        gradient_penalty = self.compute_gradient_penalty(real_data.data, fake_data.data)
        
        c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty
        c_loss.backward()
        self.optimizer_C.step()
        
        # Train Generator
        g_loss = torch.tensor(0.0)
        if torch.rand(1).item() < 0.2: # 5 critic steps per generator step
            self.optimizer_G.zero_grad()
            fake_data = self.generator(z)
            fake_validity = self.critic(fake_data)
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            self.optimizer_G.step()
            
        return c_loss.item(), g_loss.item()
