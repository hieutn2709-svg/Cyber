import torch
import torch.nn as nn
from transformers import RobertaModel
from torchcrf import CRF

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.q = nn.Parameter(torch.randn(hidden_dim)) 

    def forward(self, gru_output):
        e_t = torch.tanh(self.W1(gru_output) + self.W2(self.q)) 
        weights = torch.softmax(self.v(e_t), dim=1) 
        
        context_vector = gru_output * weights
        return context_vector, weights

class CyberEntRelModel(nn.Module):

    def __init__(self, num_labels, num_rel_types, model_config):
        super(CyberEntRelModel, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained(model_config['encoder'])
        roberta_hidden_size = 768 # Chuẩn của roberta-base

        self.bigru = nn.GRU(
            input_size=roberta_hidden_size,
            hidden_size=model_config['bigru']['dimension'], # 250
            num_layers=model_config['bigru']['num_layers'], # 2
            bidirectional=True,
            batch_first=True,
            dropout=model_config['bigru']['dropout'] # 0.5
        )
        
        gru_hidden_dim = model_config['bigru']['dimension'] * 2 

        self.attention = AttentionLayer(gru_hidden_dim)

        self.fc = nn.Linear(gru_hidden_dim, model_config['hidden_layer_neurons']) # 1536
        self.dropout = nn.Dropout(model_config['bigru']['dropout'])
        self.layer_norm = nn.LayerNorm(model_config['hidden_layer_neurons'])

        self.rel_classifier = nn.Linear(model_config['hidden_layer_neurons'], num_rel_types)

        self.crf_classifier = nn.Linear(model_config['hidden_layer_neurons'], num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None, rel_labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state 

        gru_output, _ = self.bigru(sequence_output) 
        context_vector, _ = self.attention(gru_output)

        dense_output = self.dropout(torch.relu(self.fc(context_vector)))
        dense_output = self.layer_norm(dense_output)

        logits = self.crf_classifier(dense_output)

        pooled_output, _ = torch.max(dense_output, dim=1)
        rel_logits = self.rel_classifier(pooled_output)

        if labels is not None:
            ner_loss = -self.crf(logits, labels, mask=attention_mask.bool(), reduction='token_mean')
            
            rel_loss_fct = nn.CrossEntropyLoss()
            rel_loss = rel_loss_fct(rel_logits, rel_labels)
            
            total_loss = (0.8 * ner_loss) + (0.2 * rel_loss)
            return total_loss
        else:
            prediction = self.crf.decode(logits, mask=attention_mask.bool())
            rel_prediction = torch.argmax(rel_logits, dim=-1)
            return prediction, rel_prediction