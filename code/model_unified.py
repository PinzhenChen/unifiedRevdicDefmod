import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import data
from baseline_models import PositionalEncoding

# for reproducibility
random.seed(0)
torch.manual_seed(0)

# Hyperparameters
WORD_EMBEDDING_DIM = 256  # as provided in models.py
MAX_LEN = AUTOENCODING_DIM = 256  # max length of definitions & size of autoencodings
LINEAR_DROPOUT = 0.2  # dropout of word embeddings, autoencodings, and between linear layers.

AE_TO_EMB_DEPTH = 1  # the depth of autoencoding -> word embedding
AE_DEPTH = 1  # the depth of the autoencoding -> autoencoding
EMB_TO_AE_DEPTH = 1  # the depth of word embedding -> autoencoding

# probability that an input token is corrupted, when training the autoencoder.
DEFINITION_WORD_DROPOUT = 0  # also tried 0.1
# controls what a word is corrupted to. Should be either "mask" or "random".
CORRUPTION_WORD = ["mask", "random"][0]  # unused

# transformer configs are copied from models.py which is defined by the organisers.
TRANSFORMER_DEPTH = 4  # tried 4 and 6
TRANSFORMER_NUM_HEAD = 4  # tried 4 and 8
TRANSFORMER_DROPOUT = 0.3  # tried 0.3 and 0.1


class SharedAutoEncoder(nn.Module):
    """An autoencoder, input from *2AutoEncoding and output to AutoEncoding2*.
        Transforms both encoded word embeddings and definitions using shared weights.
    """
    def __init__(self,
                 dim_in=AUTOENCODING_DIM,
                 dim_hidden=None,
                 dim_out=AUTOENCODING_DIM,
                 depth=AE_DEPTH,
                 dropout=LINEAR_DROPOUT,
                 skip=False):

        super(SharedAutoEncoder, self).__init__()

        self.skip = skip

        if dim_hidden is None:
            dim_hidden = dim_in #// 2  # //2 for compression

        layers = []
        for d in range(depth):
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(in_features=dim_in if d == 0 else dim_hidden,
                                    out_features=dim_hidden,
                                    bias=True)
                          )

        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=dim_hidden, out_features=dim_out, bias=True))

        self.architecture = nn.Sequential(*layers)

    def forward(self, x):
        if self.skip:
            x += self.architecture(x)
        else:
            x = self.architecture(x)
        return x


class Emb2AutoEncoding(nn.Module):
    """A liner layer that transforms a word embedding into an autoencoding"""
    def __init__(self,
                 depth=EMB_TO_AE_DEPTH,
                 dim_in=WORD_EMBEDDING_DIM,
                 dim_hidden=None,
                 dim_out=AUTOENCODING_DIM,
                 dropout=LINEAR_DROPOUT):

        super(Emb2AutoEncoding, self).__init__()

        if dim_hidden is None:
            dim_hidden = dim_in

        layers = []
        for d in range(depth):
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(in_features=dim_in if d == 0 else dim_hidden,
                                    out_features=dim_out if d == depth - 1 else dim_hidden,
                                    bias=True)
                          )

        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=dim_out, out_features=dim_out, bias=True))

        self.architecture = nn.Sequential(*layers)

    def forward(self, x):
        return self.architecture(x)


class AutoEncoding2Emb(nn.Module):
    """A liner layer that transforms an autoencoding into a word embedding"""
    def __init__(self,
                 depth=AE_TO_EMB_DEPTH,
                 dim_in=AUTOENCODING_DIM,
                 dim_hidden=None,
                 dim_out=WORD_EMBEDDING_DIM,
                 dropout=LINEAR_DROPOUT):

        super(AutoEncoding2Emb, self).__init__()

        if dim_hidden is None:
            dim_hidden = dim_in

        layers = []
        for d in range(depth):
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(in_features=dim_in if d == 0 else dim_hidden,
                                    out_features=dim_out if d == depth - 1 else dim_hidden,
                                    bias=True)
                          )

        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=dim_out, out_features=dim_out, bias=True))

        self.architecture = nn.Sequential(*layers)

    def forward(self, x):
        return self.architecture(x)


class Definition2AutoEncoding(nn.Module):
    """A transformer encoder that transforms a word definition into an autoencoding"""
    def __init__(self,
                 vocab,
                 dim_in,
                 dim_out,
                 n_layers,
                 n_heads,
                 definition_word_dropout,
                 corruption_word,
                 transformer_dropout,
                 max_len):

        super(Definition2AutoEncoding, self).__init__()

        self.definition_word_dropout = definition_word_dropout
        self.corruption_word = corruption_word
        self.padding_idx = vocab[data.PAD]
        self.eos_idx = vocab[data.EOS]
        self.special_vocab = {vocab[data.PAD], vocab[data.EOS], vocab[data.BOS], vocab[data.UNK]}
        self.candidate_vocab = [vocab[key] for key in list(set(list(vocab)) - self.special_vocab)]
        self.embedding = nn.Embedding(len(vocab), dim_in, padding_idx=self.padding_idx)
        self.positional_encoding = PositionalEncoding(
            dim_in, dropout=transformer_dropout, max_len=max_len
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim_in,
                                       nhead=n_heads,
                                       dropout=transformer_dropout,
                                       dim_feedforward=dim_in * 2),
            num_layers=n_layers
        )

        self.linear_layer = nn.Linear(in_features=dim_in, out_features=dim_out, bias=True)

    def corrupt_word(self, tensor):
        if not self.training or self.definition_word_dropout <= 0:
            return tensor

        corruption_mask = torch.rand_like(tensor, dtype=torch.float64)
        if self.corruption_word == "mask":
            return tensor.masked_fill(corruption_mask < self.definition_word_dropout,
                                      self.padding_idx)
        elif self.corruption_word == "random":
            return tensor.masked_fill(corruption_mask < self.definition_word_dropout,
                                      random.choice(self.candidate_vocab))
        else:
            raise NotImplementedError

    def forward(self, gloss_tensor):
        src_key_padding_mask = gloss_tensor == self.padding_idx
        corrupted_gloss_tensor = self.corrupt_word(gloss_tensor)

        embs = self.embedding(corrupted_gloss_tensor)
        src = self.positional_encoding(embs)
        transformer_output = self.transformer_encoder(src,
                                                      src_key_padding_mask=src_key_padding_mask.t())

        summed_embs = transformer_output.masked_fill(
            src_key_padding_mask.unsqueeze(-1), 0
        ).sum(dim=0)

        return self.linear_layer(F.relu(summed_embs))


class AutoEncoding2Definition(nn.Module):
    """A transformer encoder that transforms an autoencoding into a word definition"""
    def __init__(self,
                 vocab,
                 dim_in,
                 dim_out,
                 n_layers,
                 n_heads,
                 transformer_dropout,
                 dropout,
                 max_len):
        super(AutoEncoding2Definition, self).__init__()

        self.max_len = max_len
        self.padding_idx = vocab[data.PAD]
        self.eos_idx = vocab[data.EOS]
        self.positional_encoding = PositionalEncoding(
            dim_out, dropout=transformer_dropout, max_len=self.max_len
        )

        self.dropout_layer = nn.Dropout(p=dropout)
        self.linear_layer = nn.Linear(in_features=dim_in, out_features=dim_out, bias=True)
        self.embedding = nn.Embedding(len(vocab), dim_out, padding_idx=self.padding_idx)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=dim_out,
                                                     nhead=n_heads,
                                                     dropout=transformer_dropout,
                                                     dim_feedforward=dim_out * 2),
            num_layers=n_layers
        )

        self.v_proj = nn.Linear(dim_out, len(vocab))

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, x, input_sequence=None):
        device = next(self.parameters()).device
        x = self.linear_layer(self.dropout_layer(x))
        embs = self.embedding(input_sequence)
        seq = torch.cat([x.unsqueeze(0), embs], dim=0)
        src = self.positional_encoding(seq)
        src_key_padding_mask = torch.cat(
            [
                torch.tensor([[False] * input_sequence.size(1)]).to(device),
                (input_sequence == self.padding_idx),
            ],dim=0,).t()

        src_mask = self.generate_square_subsequent_mask(src.size(0)).to(device)
        transformer_output = self.transformer_encoder(
            src=src, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        return self.v_proj(transformer_output)

    @torch.no_grad()
    def pred(self, vector, decode_fn=None, beam_size=6, verbose=False):
        # which device we should cast our variables to
        device = next(self.parameters()).device

        # how many examples are batched together
        batch_size = vector.size(0)

        # Tensors will have this shape:
        # [Sequence, Batch, Beam, Continuation, *]

        # accumulation variable, keeping track of the best beams for each batched example
        generated_symbols = torch.zeros(0, batch_size, beam_size, dtype=torch.long).to(device)

        # which beams hold a completed sequence
        current_beam_size = 1
        has_stopped = torch.tensor([False] * (batch_size * current_beam_size)).to(device)

        # the input to kick-start the generation is the embedding, we start with the same input for each beam
        vector_src = vector.unsqueeze(1).expand(batch_size, current_beam_size, -1).reshape(1,  batch_size * current_beam_size, -1)
        vector_src = self.linear_layer(vector_src)
        src = vector_src
        src_key_padding_mask = torch.tensor([[False] * (batch_size * current_beam_size)]).to(device)

        # variables needed to compute the score of each beam (geometric mean of probability of emission)
        logprobs = torch.zeros(batch_size, current_beam_size, dtype=torch.double).to(device)
        lengths = torch.zeros(batch_size * current_beam_size, dtype=torch.int).to(device)
        # generate tokens step by step
        for step_idx in range(self.max_len):

            # generation mask
            src_mask = self.generate_square_subsequent_mask(src.size(0)).to(device)
            # positional encoding
            src_pe = self.positional_encoding(src)
            # transformer output
            transformer_output = self.transformer_encoder(
                src_pe, mask=src_mask, src_key_padding_mask=src_key_padding_mask.t()
            )[-1]
            # distribution over the full vocabulary
            v_dist = self.v_proj(transformer_output)
            # don't generate padding tokens
            v_dist[...,self.padding_idx] = -float("inf")
            v_dist = F.log_softmax(v_dist, dim=-1)

            # for each beam, select the best candidate continuations
            new_logprobs, new_symbols = v_dist.topk(beam_size, dim=-1)
            # patch the output scores to zero-out items that have already stopped
            new_logprobs = new_logprobs.masked_fill(has_stopped.unsqueeze(-1), 0.0)
            # if the beam hasn't stopped, then it needs to produce at least an EOS
            # so we can just add one to beams that have not stopped to account for the current token
            lengths += (~has_stopped).int()

            # compute scores for each continuation
            ## recreate the score of the previous full sequence for all possible continuations
            logprobs_ = logprobs.view(batch_size * current_beam_size, 1).expand(batch_size * current_beam_size, beam_size)
            ## add the cost of each continuation
            logprobs_ = logprobs_ + new_logprobs
            ## average over the full sequence, ignoring padding items
            avg_logprobs = logprobs_ #/ lengths.unsqueeze(-1)
            ## select the `beam_size` best continuations overall, their matching scores will be `avg_logprobs`
            avg_logprobs, selected_beams = avg_logprobs.view(batch_size, current_beam_size * beam_size).topk(beam_size, dim=-1)
            ## select back the base score for the selected continuations
            logprobs = logprobs_.view(batch_size, current_beam_size * beam_size).gather(-1, selected_beams).view(batch_size, beam_size)

            # add symbols of best continuations
            ## recreate the full previous sequence for all possible continuations
            generated_symbols_ = generated_symbols.view(-1, batch_size * current_beam_size, 1).expand(-1, batch_size * current_beam_size, beam_size)
            ## stack on the new symbols
            generated_symbols_ = torch.cat([generated_symbols_, new_symbols.unsqueeze(0)], dim=0)
            ## grab only the `beam_size` best continuations out of all possible continuations
            generated_symbols_ = generated_symbols_.view(-1, batch_size, current_beam_size * beam_size)
            generated_symbols = generated_symbols_.gather(-1, selected_beams.unsqueeze(0).expand(step_idx + 1, batch_size,  beam_size)).view(step_idx + 1, batch_size, beam_size)

            # recompute which beams have stopped, and what their lengths are
            ## reconstruct the lengths of all candidate continuations
            lengths = lengths.view(batch_size, current_beam_size, 1).expand(batch_size, current_beam_size, beam_size)
            ## retrieve the lengths of the selected beam continuations
            lengths = lengths.reshape(batch_size, current_beam_size * beam_size).gather(-1, selected_beams).view(-1)
            ## reconstruct the halting state of all candidate continuations
            has_stopped = has_stopped.view(batch_size, current_beam_size, 1).expand(batch_size, current_beam_size, beam_size)
            ## retrieve the halting states of selected beam continuations
            has_stopped = has_stopped.reshape(batch_size, current_beam_size * beam_size).gather(-1, selected_beams).view(-1)

            # flag which beams have terminated at the current step (i.e., whether they just produced an EOS)
            generated_symbols = generated_symbols.view(-1, batch_size * beam_size)
            generated_symbols[-1] = generated_symbols[-1].masked_fill(has_stopped, self.padding_idx)
            has_stopped = has_stopped | (generated_symbols.view(-1, batch_size * beam_size)[-1] == self.eos_idx).view(batch_size * beam_size)

            # recompute padding mask on the basis of which continuations were selected
            src_key_padding_mask = src_key_padding_mask.view(-1, batch_size, current_beam_size, 1).expand(-1, batch_size, current_beam_size, beam_size)
            src_key_padding_mask = src_key_padding_mask.reshape(-1, batch_size, current_beam_size * beam_size)
            src_key_padding_mask = src_key_padding_mask.gather(-1, selected_beams.unsqueeze(0).expand(step_idx + 1, batch_size,  beam_size)).view(step_idx + 1, batch_size * beam_size)
            src_key_padding_mask = torch.cat([src_key_padding_mask, has_stopped.unsqueeze(0)], dim=0)

            # produce input for the next timestep
            src = torch.cat([vector_src.expand(1, beam_size, -1), self.embedding(generated_symbols)], dim=0)
            # reshape to the familiar format
            generated_symbols = generated_symbols.view(-1, batch_size, beam_size)

            # if all beams have stopped, so do we
            if has_stopped.all():
                break
            # we update the number of sustained beam at the first iteration, since we know have `beam_size` candidates.
            current_beam_size = beam_size

        # select the most likely sequence for each batched item
        max_scores, selected_beams = (logprobs / lengths.view(batch_size, beam_size)).topk(1, dim=1)
        output_sequence = generated_symbols.gather(1, selected_beams.unsqueeze(0).expand(step_idx + 1, batch_size, 1))

        if verbose:
            print(decode_fn(output_sequence.squeeze(-1)))

        return output_sequence.squeeze(-1)


class OurModel(nn.Module):
    """Our autoencoder model"""

    def __init__(self,
                 vocab,
                 emb_to_ae_depth=EMB_TO_AE_DEPTH,
                 ae_depth=AE_DEPTH,
                 ae_to_emb_depth=AE_TO_EMB_DEPTH,
                 word_emb_dim=WORD_EMBEDDING_DIM,
                 ae_dim=AUTOENCODING_DIM,
                 dim_hidden=None,
                 dropout=LINEAR_DROPOUT,
                 transformer_dropout=TRANSFORMER_DROPOUT,
                 transformer_depth=TRANSFORMER_DEPTH,
                 transformer_num_head=TRANSFORMER_NUM_HEAD,
                 definition_word_dropout=DEFINITION_WORD_DROPOUT,
                 corruption_word=CORRUPTION_WORD,
                 max_len=MAX_LEN,
                 skip=False
                 ):
        super(OurModel, self).__init__()

        self.padding_idx = vocab[data.PAD]
        self.max_len = max_len

        self.shared_ae = SharedAutoEncoder(
            dim_in=ae_dim,
            dim_hidden=dim_hidden,
            dim_out=ae_dim,
            depth=ae_depth,
            dropout=dropout,
            skip=skip
        )

        self.word_embed_to_ae = Emb2AutoEncoding(
            depth=emb_to_ae_depth,
            dim_in=word_emb_dim,
            dim_hidden=dim_hidden,
            dim_out=ae_dim,
            dropout=dropout
        )

        self.ae_to_word_embed = AutoEncoding2Emb(
            depth=ae_to_emb_depth,
            dim_in=ae_dim,
            dim_hidden=dim_hidden,
            dim_out=word_emb_dim,
            dropout=dropout
        )

        self.def_to_ae = Definition2AutoEncoding(
            vocab,
            dim_in=word_emb_dim,
            dim_out=ae_dim,
            n_layers=transformer_depth,
            n_heads=transformer_num_head,
            definition_word_dropout=definition_word_dropout,
            corruption_word=corruption_word,
            transformer_dropout=transformer_dropout,
            max_len=max_len
        )

        self.ae_to_def = AutoEncoding2Definition(
            vocab,
            dim_in=ae_dim,
            dim_out=word_emb_dim,
            n_layers=transformer_depth,
            n_heads=transformer_num_head,
            transformer_dropout=transformer_dropout,
            dropout=dropout,
            max_len = max_len
        )

    def forward(self, word_embedding, definition, mode="both"):

        if mode == "inference":  # def->embed
            if word_embedding is None and definition is not None:
                x = self.def_to_ae(definition)
                x = self.shared_ae(x)
                embed_conv = self.ae_to_word_embed(x)
                return None, None, None, None, embed_conv, None

            elif definition is not None and word_embedding is not None:  # embed->def
                x = self.word_embed_to_ae(word_embedding)
                x = self.shared_ae(x)
                def_conv = self.ae_to_def(x, definition[:-1])
                return None, None, None, None, None, def_conv

            else:
                raise NotImplementedError

        else:
            ae_word = self.shared_ae(self.word_embed_to_ae(word_embedding))
            ae_def = self.shared_ae(self.def_to_ae(definition))

            if mode == "both":  # train reconstruction and conversion at the same time
                embed_recon = self.ae_to_word_embed(ae_word)
                def_recon = self.ae_to_def(ae_def, definition[:-1])
                embed_conv = self.ae_to_word_embed(ae_def)
                def_conv = self.ae_to_def(ae_word, definition[:-1])
                return ae_word, ae_def, embed_recon, def_recon, embed_conv, def_conv

            elif mode == "reconstruct":
                embed_recon = self.ae_to_word_embed(ae_word)
                def_recon = self.ae_to_def(ae_def, definition[:-1])
                return ae_word, ae_def, embed_recon, def_recon, None, None

            elif mode == "convert":
                embed_conv = self.ae_to_word_embed(ae_def)
                def_conv = self.ae_to_def(ae_word, definition[:-1])
                return ae_word, ae_def, None, None, embed_conv, def_conv

            else:
                raise NotImplementedError

    @torch.no_grad()
    def pred(self, word_embedding, decode_fn=None, beam_size=6, verbose=False):
        x = self.shared_ae(self.word_embed_to_ae(word_embedding))
        return self.ae_to_def.pred(x,
                                   decode_fn=decode_fn,
                                   beam_size=beam_size,
                                   verbose=verbose)

    @staticmethod
    def load(file, map_location):
        return torch.load(file, map_location=map_location)

    def save(self, file):
        torch.save(self, file)

