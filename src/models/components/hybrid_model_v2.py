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

class UserRatingEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.embeding = nn.Sequential(
            nn.Linear(31, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

    def forward(self, x):
        x = torch.cat([x[0], x[1], x[2]], dim=1)
        x = self.embeding(x)
        return x

class UserRatingEmbeddingV2(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.general_embeding = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        self.age_embeding = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        self.occupation_embeding = nn.Sequential(
            nn.Linear(23, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        self.embeding = nn.Sequential(
            nn.Linear(128 * 3, 256),
        )

    def forward(self, x):
        
        x = torch.cat([x[0], x[1], x[2]], dim=1)
        x = self.embeding(x)
        return x

class HybridModelV2(nn.Module):
    def __init__(self, text_embedding=TextEmbedding(), image_embedding=ImageEmbedding(), user_rating_embedding=UserRatingEmbedding(), num_classes=18):
        super().__init__()
        self.text_embedding = text_embedding
        self.image_embedding = image_embedding
        self.user_rating_embedding = user_rating_embedding
        self.classifier = nn.Sequential(
            nn.Linear(768 + 4096 + 256, 768 + 4096 + 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(768 + 4096 + 256, num_classes)  # Replace last FC layer with num_classes output
        )

    def forward(self, user_rating, text_input_ids, text_attention_mask, image_input):
        user_rating_embedding = self.user_rating_embedding(user_rating)
        text_embedding = self.text_embedding(text_input_ids, text_attention_mask)
        image_embedding = self.image_embedding(image_input)
        
        combined_embedding = torch.cat((text_embedding, image_embedding.view(image_embedding.size(0), -1), user_rating_embedding), dim=1)
        logits = self.classifier(combined_embedding)
        
        return torch.sigmoid(logits)

# if __name__ == "__main__":
#     hybrid_model = HybridModelV2()
#     ratings_general = torch.randn(16, 3)
#     ratings_age = torch.randn(16, 7)
#     ratings_occupation = torch.randn(16, 20)
#     title_input_ids = torch.ones([16, 23]).long()
#     title_attn_mask = torch.ones([16, 23]).long()
#     img_tensor = torch.zeros([16, 3, 224, 224], dtype=torch.float32)
#     out = hybrid_model((ratings_general, ratings_age, ratings_occupation), title_input_ids, title_attn_mask, img_tensor)
    
#     print(out.shape)

############################################################### TEST ###############################################################
import hydra
from omegaconf import DictConfig
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
@hydra.main(version_base="1.3", config_path="../../../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    # test instance
    dm = hydra.utils.instantiate(cfg.data)
    dm.prepare_data()
    dm.setup()
    
    # test dataset
    print(f"Length of train dataset: {len(dm.data_train)}")
    print(f"Length of val dataset: {len(dm.data_val)}")
    print(f"Length of test dataset: {len(dm.data_test)}\n")
    
    # test dataloader
    print(f"Length of train dataloader: {len(dm.train_dataloader())}")
    print(f"Length of val dataloader: {len(dm.val_dataloader())}")
    print(f"Length of test dataloader: {len(dm.test_dataloader())}\n")
    
    # test batch
    (ratings_general, ratings_age, ratings_occupation), title_input_ids, title_attn_mask, img_tensor, genre_tensor = next(iter(dm.val_dataloader()))
    
    print(f"Length of one batch (batch size): {len(genre_tensor)}")
    print(f"Shape of ratings_general: {ratings_general.size()}")
    print(f"Shape of ratings_age: {ratings_age.size()}")
    print(f"Shape of ratings_occupation: {ratings_occupation.size()}")
    print(f"Shape of title_input_ids: {title_input_ids.size()}")
    print(f"Shape of title_attn_mask: {title_attn_mask.size()}")
    print(f"Shape of img_tensor: {img_tensor.size()}")
    print(f"Shape of output: {genre_tensor.size()}")
    
    hybrid_model = HybridModelV2()
    out = hybrid_model((ratings_general, ratings_age, ratings_occupation), title_input_ids, title_attn_mask, img_tensor)
    print(out.shape)

if __name__ == "__main__":
    main()
############################################################### TEST ###############################################################
    