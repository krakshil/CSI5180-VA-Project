import torch
import torch.optim as optim
import numpy as np

from allennlp_models.rc.models.bidaf import BidirectionalAttentionFlow
from allennlp.data import DataLoader, DatasetReader
from allennlp.data.dataset_readers import SquadReader
from allennlp.training.trainer import Trainer
from allennlp.models import load_archive
from allennlp.predictors import Predictor
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import SquadReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.modules import TextFieldEmbedder, TimeDistributed, FeedForward
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, Seq2SeqEncoder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.similarity_functions.dot_product import DotProductSimilarity
from allennlp.training.trainer import Trainer
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.nn.util import get_text_field_mask
from allennlp.models import Model
from allennlp.modules.token_embedders import Embedding

# Define the data reader and iterator
reader = SquadReader()
train_data = reader.read("train-v2.0.json")
validation_data = reader.read("dev-v2.0.json")
token_indexer = PretrainedTransformerIndexer("bert-base-uncased")
train_iterator = BucketIterator(batch_size=32, sorting_keys=[("passage", "num_tokens")])
train_iterator.index_with(Vocabulary())

# Define the model and optimizer
text_field_embedder = BasicTextFieldEmbedder({"tokens": PretrainedTransformerEmbedder("bert-base-uncased", requires_grad=False)})
encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=text_field_embedder.get_output_dim(),
                                              hidden_size=128,
                                              num_layers=2,
                                              batch_first=True,
                                              bidirectional=True))

similarity_function = DotProductSimilarity()

modeling_layer = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=encoder.get_output_dim()*4,
                                                     hidden_size=128,
                                                     num_layers=2,
                                                     batch_first=True,
                                                     bidirectional=True))

span_end_encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=encoder.get_output_dim()*2,
                                                       hidden_size=128,
                                                       num_layers=2,
                                                       batch_first=True,
                                                       bidirectional=True))
model = BidirectionalAttentionFlow(vocab=Vocabulary(),
              text_field_embedder=text_field_embedder,
              encoder=encoder,
              similarity_function=similarity_function,
              modeling_layer=modeling_layer,
              span_end_encoder=span_end_encoder)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the trainer and start training
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=train_iterator,
                  train_dataset=train_data,
                  validation_dataset=validation_data,
                  patience=10,
                  num_epochs=20)
trainer.train()

# Save the trained model
archive = trainer.create_serialization_archive("model.tar.gz")
