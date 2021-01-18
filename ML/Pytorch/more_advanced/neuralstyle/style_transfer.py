import os
import torch
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image


class StyleTransferNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.chosen_features = ["0", "5", "10", "19", "28"]
        self.model = models.vgg19(pretrained=True).features
        # fix parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if str(i) in self.chosen_features:
                features.append(x)
        return features


def load_images(device, img_size):
    my_transforms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    original_img = Image.open(os.path.dirname(__file__) + "/annahathaway.png")
    style_img = Image.open(os.path.dirname(__file__) + "/style.png")
    original_img = my_transforms(original_img).unsqueeze(0).to(device)
    style_img = my_transforms(style_img).unsqueeze(0).to(device)
    return original_img, style_img


def image_style_transfer(original_img, style_img, device):
    generated = original_img.clone().requires_grad_(True)
    model = StyleTransferNet().to(device)

    # Hyperparameters
    total_steps = 1000
    learning_rate = 0.01
    alpha = 1
    beta = 0.1

    # note that we are not going to adjust model parameters but the input image
    optimizer = optim.Adam([generated], lr=learning_rate)

    for step in range(total_steps):
        generated_features = model(generated)
        original_img_features = model(original_img)
        style_features = model(style_img)

        # Loss is 0 initially
        style_loss = 0
        original_loss = 0

        # iterate through all the features for the chosen layers
        for gen_feature, orig_feature, style_feature in zip(generated_features, original_img_features, style_features):
            _, channel, height, width = gen_feature.shape

            original_loss += torch.mean((gen_feature - orig_feature) ** 2)

            # Compute Gram Matrix of generated
            G = gen_feature.view(channel, height * width).mm(
                gen_feature.view(channel, height * width).t()
            )
            # Compute Gram Matrix of Style
            A = style_feature.view(channel, height * width).mm(
                style_feature.view(channel, height * width).t()
            )
            style_loss += torch.mean((G - A) ** 2)

        total_loss = alpha * original_loss + beta * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 50 == 49:
            print(total_loss)
            save_image(generated, os.path.dirname(__file__) + f"/generated_{step}.png")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_img, style_img = load_images(device, img_size=356)
    image_style_transfer(original_img, style_img, device)
