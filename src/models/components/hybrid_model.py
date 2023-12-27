import torch
import torch.nn as nn
from transformers import BertModel
import torchvision.models as models
from torchsummary import summary

class TextEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        self.embeding = nn.Sequential(
            nn.Linear(768, 768),  # Replace first FC layer
            nn.ReLU(True),
            nn.Dropout(),
        )

        # unfreezes the last freeze_bert_layers blocks of BERT
        # freeze_bert_layers = 2
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in list(self.bert.parameters())[-(4 * 12):]:
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        # outputs = self.bert(input_ids, attention_mask=attention_mask)
        # return outputs['pooler_output']
        
        token_embedding, sentence_embedding = self.bert(input_ids, attention_mask)[:2]
        out = self.embeding(sentence_embedding)
        return out

class ImageEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        vgg19_bn = models.vgg19_bn(pretrained=True)
        
        self.features = vgg19_bn.features
        for param in self.features.parameters():
            param.requires_grad = False
            
        self.embeding = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # Replace first FC layer
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.embeding(x)
        return x

class HybridModel(nn.Module):
    def __init__(self, text_embedding=TextEmbedding(), image_embedding=ImageEmbedding(), num_classes=18):
        super().__init__()
        self.text_embedding = text_embedding
        self.image_embedding = image_embedding
        self.classifier = nn.Sequential(
            nn.Linear(4864, 4864),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4864, num_classes)  # Replace last FC layer with num_classes output
        )

    def forward(self, text_input_ids, text_attention_mask, image_input):
        text_embedding = self.text_embedding(text_input_ids, text_attention_mask)
        image_embedding = self.image_embedding(image_input)
        combined_embedding = torch.cat((text_embedding, image_embedding.view(image_embedding.size(0), -1)), dim=1)
        logits = self.classifier(combined_embedding)
        
        return torch.sigmoid(logits)

if __name__ == "__main__":
    hybrid_model = HybridModel()
    title_input_ids = torch.ones([16, 23]).long()
    title_attn_mask = torch.ones([16, 23]).long()
    img_tensor = torch.zeros([16, 3, 224, 224], dtype=torch.float32)
    out = hybrid_model(title_input_ids, title_attn_mask, img_tensor)
    