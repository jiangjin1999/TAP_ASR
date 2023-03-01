from torch import nn
class audio_encoder(nn.Module):
    def __init__(self, mlp_dim: int=768, fc_output_dim: int = 512):
        super(audio_encoder, self).__init__()

        self.fc= nn.Linear(mlp_dim, fc_output_dim)

    def forward(self, batch):
        output = self.fc(batch)

        return output