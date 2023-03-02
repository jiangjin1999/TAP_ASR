from torch import nn
class audio_encoder(nn.Module):
    def __init__(self, mlp_dim: int=768, fc_output_dim: int = 512):
        super(audio_encoder, self).__init__()

        self.fc_1 = nn.Linear(mlp_dim, fc_output_dim)
        self.fc_2 = nn.Linear(fc_output_dim, fc_output_dim)
        self.relu= nn.ReLU()

    def forward(self, batch):
        output = self.fc_2(self.relu(self.fc_1(batch)))

        return output