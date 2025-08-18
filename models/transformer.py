# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, Literal, Callable

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: Tensor, mask: Tensor, query_embed: Tensor, pos_embed: Tensor):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
    

class SpatialTemporalTransformer(Transformer):
    def __init__(self, num_frames: int, enc_time_attn: Literal["joint", "div", "none"] = "none", dec_time_attn: bool = False, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, 
                 dim_feedforward=2048, dropout=0.1, activation="relu", return_intermediate_dec=False):
        super().__init__(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, 
                         dropout, activation, False, return_intermediate_dec)
        self.num_frames = num_frames
        self.enc_time_attn = enc_time_attn
        self.dec_time_attn = dec_time_attn

        if enc_time_attn == "none" or num_frames <= 1:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, False)
        else:
            if enc_time_attn == "joint":
                encoder_layer = JointSpaceTimeEncoderLayer(d_model, nhead, num_frames, dim_feedforward,
                                                    dropout, activation, False)
            elif enc_time_attn == "div":
                encoder_layer = DividedSpaceTimeEncoderLayer(d_model, nhead, num_frames, dim_feedforward,
                                                    dropout, activation, False)
                
        if dec_time_attn and num_frames > 1:
            decoder_layer = SpaceTimeDecoderLayer(d_model, nhead, num_frames, dim_feedforward, dropout, activation)
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, False)
        decoder_norm = nn.LayerNorm(d_model)
                
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

    def _encoder_reshape_tensors(self, src: Tensor, mask: Tensor, query_embed: Tensor, pos_embed: Tensor, time_pos_embed: Tensor = None, query_time_embed: Tensor = None):
        # flatten NxCxHxW to HWxNxC
        bsxt, c, h, w = src.shape
        assert bsxt % self.num_frames == 0, "batch size: {}, num_frames: {}".format(bsxt, self.num_frames)

        if self.enc_time_attn in ["none", "div"] or self.num_frames <= 1: # do not use temporal attn | divided attn
            src = src.flatten(2).permute(2, 0, 1) # [HW, BxT, D]
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1) # [HW, BxT, D]
            mask = mask.flatten(1) # [BxT, HW]
            if self.enc_time_attn == "div": # [T, 1] -> [T, BxHW, D]
                time_pos_embed = time_pos_embed.unsqueeze(-1).repeat(1, bsxt // self.num_frames * h * w, c)
        else: # joint sapce-time attn
            _couple: Callable[[Tensor], Tensor] = lambda x: x.flatten(2).unflatten(0, [-1, self.num_frames]).permute(1, 3, 0, 2).flatten(0, 1) # [BxT, D, H, W] -> [TxHW, B, D]
            src, pos_embed = _couple(src), _couple(pos_embed) # [TxHW, B, D]
            mask = mask.unflatten(0, [-1, self.num_frames]).flatten(1) # [BxT, H, W] -> [B, TxHW]
            if time_pos_embed is not None:  # [T, 1] -> [TxHW, B, D]
                time_pos_embed = time_pos_embed.unsqueeze(-1).repeat(h * w, bsxt // self.num_frames, c)
                pos_embed += time_pos_embed

        query_embed = query_embed.unsqueeze(1).repeat(1, bsxt // self.num_frames, 1) # [N, B, D]
        if self.dec_time_attn and self.num_frames > 1:
            query_embed = query_embed.repeat(1, self.num_frames, 1) # [N, BxT, D]
            # [T, 1] -> [T, BxN, D]
            query_time_embed = query_time_embed.unsqueeze(-1).repeat(1, bsxt // self.num_frames * query_embed.size(0), c)

        return src, mask, query_embed, pos_embed, time_pos_embed, query_time_embed
    
    def _decoder_reshape_tensors(self, memory: Tensor, pos_embed: Tensor, mask: Tensor):
        _view: Callable[[Tensor], Tensor] = lambda x: x
        if self.enc_time_attn in ["none", "div"] or self.num_frames <= 1: # memory shape [HW, BxT, D]
            if not self.dec_time_attn:
                _view = lambda x: x.unflatten(1, [-1, self.num_frames]).permute(2, 0, 1, 3).flatten(0, 1) # [HW, BxT, D] -> [TxHW, B, D]
                mask = mask.unflatten(0, [-1, self.num_frames]).flatten(1) # [B, TxHW]
        else: # joint space-time attn, memory shape: [TxHW, B, D]
            if self.dec_time_attn:
                _view = lambda x: x.unflatten(0, [self.num_frames, -1]).permute(1, 2, 0, 3).flatten(1, 2) # [HW, BxT, D]
                mask = mask.unflatten(1, [self.num_frames, -1]).flatten(0, 1) # [BxT, HW]

        memory, pos_embed = _view(memory), _view(pos_embed)

        return memory, pos_embed, mask
    
    def _reshape_memory(self, memory: Tensor, fH: int, fW: int):
        if self.dec_time_attn:
            # [HW, BxT, D] -> [H, W, B, T, D] -> [B, T, D, H, W] -> [BxT, D, H, W]
            return memory.view(fH, fW, -1, self.num_frames, memory.size(-1)).permute(2, 3, 4, 0, 1).flatten(0, 1)
        else: 
            # [TxHW, B, D] -> [T, H, W, B, D] -> [B, T, D, H, W] -> [BxT, D, H, W]
            return memory.unflatten(0, [self.num_frames, fH, fW]).permute(3, 0, 4, 1, 2).flatten(0, 1)


    def forward(self, src: Tensor, mask: Tensor, query_embed: Tensor, pos_embed: Tensor, time_pos_embed: Tensor = None, query_time_embed: Tensor = None):
        h, w = src.shape[-2:]
        src, mask, query_embed, pos_embed, time_pos_embed, query_time_embed = self._encoder_reshape_tensors(src, mask, query_embed, pos_embed, time_pos_embed, query_time_embed)
        tgt = torch.zeros_like(query_embed)

        memory: Tensor = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed, time_pos=time_pos_embed)

        memory, pos_embed, mask = self._decoder_reshape_tensors(memory, pos_embed, mask) # [TxHW, B, D] or [HW, BxT, D], ..., [B, TxHW]

        hs: Tensor = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_spatial_pos=query_embed, query_temporal_pos=query_time_embed) # [N, B, D] or [N, BxT, D]
        
        return hs.transpose(1, 2), self._reshape_memory(memory, h, w)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                time_pos: Tensor = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, time_pos=time_pos)

        if self.norm is not None:
            output = self.norm(output)
            
        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_spatial_pos: Tensor = None, query_temporal_pos: Tensor = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_spatial_pos=query_spatial_pos, query_temporal_pos=query_temporal_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                time_pos: Tensor = None) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_spatial_pos: Tensor = None, query_temporal_pos: Tensor = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_spatial_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_spatial_pos)
    

class JointSpaceTimeEncoderLayer(TransformerEncoderLayer):
    def __init__(self, d_model, nhead, num_frames: int, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        self.num_frames = num_frames
    

class DividedSpaceTimeEncoderLayer(TransformerEncoderLayer):
    def __init__(self, d_model, nhead, num_frames: int, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        self.num_frames = num_frames

        self.time_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.time_dropout = nn.Dropout(dropout)
        self.time_norm = nn.LayerNorm(d_model)

    def forward(self, src: Tensor, src_mask: Tensor = None, src_key_padding_mask: Tensor = None, pos: Tensor = None, time_pos: Tensor = None):
        hw = src.size(0)
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src) # [HW, BxT, D]

        src = src.unflatten(1, [-1, self.num_frames]).permute(2, 1, 0, 3).flatten(1, 2) # [T, BxHW, D]
        q = k = self.with_pos_embed(src, time_pos)
        src2 = self.time_attn(q, k, value=src)[0]
        src = src + self.time_dropout(src2)
        src = self.time_norm(src) # [T, BxHW, D]
        src = src.unflatten(1, [-1, hw]).permute(2, 1, 0, 3).flatten(1, 2) # [HW, BxT, D]

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src # [HW, BxT, D]
    

class SpaceTimeDecoderLayer(TransformerDecoderLayer):
    def __init__(self, d_model: int, nhead: int, num_frames: int, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, False)

        self.num_frames = num_frames

        self.time_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.time_dropout = nn.Dropout(dropout)
        self.time_norm = nn.LayerNorm(d_model)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor = None, memory_mask: Tensor = None, 
                tgt_key_padding_mask: Tensor = None, memory_key_padding_mask: Tensor = None, pos: Tensor = None, 
                query_spatial_pos: Tensor = None, query_temporal_pos: Tensor = None):
        # tgt shape: [N, BxT, D]
        num_queries = tgt.size(0)
        q = k = self.with_pos_embed(tgt, query_spatial_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_spatial_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt = tgt.unflatten(1, [-1, self.num_frames]).permute(2, 1, 0, 3).flatten(1, 2) # [T, BxN, D]
        q = k = self.with_pos_embed(tgt, query_temporal_pos)
        tgt2 = self.time_attn(query=q, key=k, value=tgt)[0]
        tgt = tgt + self.time_dropout(tgt2)
        tgt = self.time_norm(tgt)
        tgt = tgt.unflatten(1, [-1, num_queries]).permute(2, 1, 0, 3).flatten(1, 2) # [N, BxT, D]

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )

def build_st_transformer(args):
    return SpatialTemporalTransformer(
        num_frames=args.num_frames,
        enc_time_attn=args.enc_time_attn,
        dec_time_attn=args.dec_time_attn,
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
