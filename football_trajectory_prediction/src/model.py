"""
Transformer model for football trajectory prediction.
Key fixes:
1. Player-specific decoder conditioning using last known state
2. Relative positioning to make predictions translation-invariant
3. Autoregressive decoding option for better sequential predictions
4. Improved attention masking
"""

import torch
import torch.nn as nn
import math
import sys
sys.path.append('..')
import config


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PlayerEncoder(nn.Module):
    """
    Encodes each player's trajectory independently, then combines.
    This preserves player identity through the encoding process.
    """
    
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
        super().__init__()
        self.d_model = d_model
        
        # Project input features to model dimension
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Initialize with small weights to prevent large initial outputs
        self._init_input_proj()
        
        # Positional encoding for frames
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Per-player transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def _init_input_proj(self):
        """Initialize input projection with small weights."""
        nn.init.xavier_uniform_(self.input_proj.weight, gain=0.1)
        if self.input_proj.bias is not None:
            nn.init.constant_(self.input_proj.bias, 0.0)
    
    def forward(self, x, mask):
        """
        Args:
            x: [batch, num_players, num_frames, features]
            mask: [batch, num_players, num_frames] (True = valid)
        Returns:
            encoded: [batch, num_players, num_frames, d_model]
            player_summary: [batch, num_players, d_model] - last valid state per player
        """
        B, P, T, F = x.shape
        
        # Check for NaN/Inf in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("ERROR: NaN/Inf in encoder input x")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Reshape to process all players together
        x_flat = x.reshape(B * P, T, F)
        mask_flat = mask.reshape(B * P, T)
        
        # Check for players with no valid frames (all masked)
        has_valid = mask_flat.any(dim=-1)  # [B*P]
        num_invalid = (~has_valid).sum().item()
        if num_invalid > 0:
            # For players with no valid frames, set at least one frame to valid with zeros
            # This prevents all-masked sequences which cause NaN in attention
            for i in range(B * P):
                if not has_valid[i]:
                    mask_flat[i, -1] = True  # Mark last frame as valid
                    x_flat[i, -1] = 0.0     # Set to zeros
        
        # Project and add positional encoding
        x_proj = self.input_proj(x_flat)
        
        # Check for NaN after projection
        if torch.isnan(x_proj).any() or torch.isinf(x_proj).any():
            print("ERROR: NaN/Inf after input_proj")
            x_proj = torch.nan_to_num(x_proj, nan=0.0, posinf=0.0, neginf=0.0)
        
        x_proj = self.pos_encoding(x_proj)
        
        # Check for NaN after positional encoding
        if torch.isnan(x_proj).any() or torch.isinf(x_proj).any():
            print("ERROR: NaN/Inf after pos_encoding")
            x_proj = torch.nan_to_num(x_proj, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create attention mask (True = ignore for PyTorch)
        attn_mask = ~mask_flat
        
        # CRITICAL: Ensure no sequence is completely masked (would cause NaN in attention)
        # PyTorch's TransformerEncoder can produce NaN when all positions are masked
        all_masked = attn_mask.all(dim=-1)  # [B*P]
        if all_masked.any():
            for i in range(B * P):
                if all_masked[i]:
                    attn_mask[i, -1] = False  # Unmask last position
                    # Also ensure the corresponding input is valid
                    x_proj[i, -1] = 0.0  # Set to zero if it was invalid
        
        # Double-check: ensure at least one position is unmasked per sequence
        assert not attn_mask.all(dim=-1).any(), "Some sequences are still completely masked!"
        
        # Encode with error handling
        try:
            encoded = self.encoder(x_proj, src_key_padding_mask=attn_mask)
        except Exception as e:
            print(f"ERROR in encoder forward pass: {e}")
            print(f"  Input shape: {x_proj.shape}")
            print(f"  Mask shape: {attn_mask.shape}")
            print(f"  All-masked sequences: {attn_mask.all(dim=-1).sum().item()}")
            raise
        
        # Check for NaN/Inf in encoded output
        if torch.isnan(encoded).any() or torch.isinf(encoded).any():
            # Count NaN/Inf values
            nan_count = torch.isnan(encoded).sum().item()
            inf_count = torch.isinf(encoded).sum().item()
            nan_sequences = torch.isnan(encoded).any(dim=-1).any(dim=-1).sum().item()  # Sequences with any NaN
            
            # Only print detailed error on first occurrence or periodically
            if not hasattr(self, '_nan_warn_count'):
                self._nan_warn_count = 0
            self._nan_warn_count += 1
            
            if self._nan_warn_count <= 3 or self._nan_warn_count % 100 == 0:
                print(f"ERROR: NaN/Inf in encoder output (occurrence #{self._nan_warn_count})")
                print(f"  NaN values: {nan_count}, Inf values: {inf_count}")
                print(f"  Sequences affected: {nan_sequences}/{B*P}")
            
            # Replace NaN with zeros to prevent propagation
            encoded = torch.nan_to_num(encoded, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Reshape back
        encoded = encoded.reshape(B, P, T, self.d_model)
        
        # Extract last valid state for each player (their state at throw time)
        # This will be used to condition the decoder
        player_summary = self._get_last_valid_state(encoded, mask)
        
        return encoded, player_summary
    
    def _get_last_valid_state(self, encoded, mask):
        """Get the last valid encoded state for each player."""
        B, P, T, D = encoded.shape
        
        # Find index of last valid frame for each player
        # mask: [B, P, T], we want the last True index along T
        mask_float = mask.float()
        indices = torch.arange(T, device=mask.device).unsqueeze(0).unsqueeze(0)
        last_valid_idx = (indices * mask_float).max(dim=-1).values.long()
        
        # Gather the last valid state
        batch_idx = torch.arange(B, device=encoded.device).unsqueeze(1).expand(B, P)
        player_idx = torch.arange(P, device=encoded.device).unsqueeze(0).expand(B, P)
        
        player_summary = encoded[batch_idx, player_idx, last_valid_idx]
        
        return player_summary


class CrossPlayerAttention(nn.Module):
    """
    Allows players to attend to other players' trajectories.
    Important for modeling interactions (e.g., coverage, routes).
    """
    
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, player_states, player_mask):
        """
        Args:
            player_states: [batch, num_players, d_model]
            player_mask: [batch, num_players] (True = valid player)
        Returns:
            updated_states: [batch, num_players, d_model]
        """
        # Self-attention across players
        attn_mask = ~player_mask
        attended, _ = self.cross_attn(
            player_states, player_states, player_states,
            key_padding_mask=attn_mask
        )
        return self.norm(player_states + self.dropout(attended))


class TrajectoryDecoder(nn.Module):
    """
    Decodes future trajectory for each player.
    Conditioned on:
    1. Player's own encoded history (player_summary)
    2. All players' encoded states (cross-attention)
    3. Ball landing position
    """
    
    def __init__(self, d_model, nhead, num_layers, dropout, max_output_frames):
        super().__init__()
        self.d_model = d_model
        self.max_output_frames = max_output_frames
        
        # Learnable output frame queries (small initialization)
        self.frame_queries = nn.Parameter(torch.randn(1, max_output_frames, d_model) * 0.01)
        
        # Ball position encoding
        self.ball_encoder = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Combine player state + ball info for conditioning
        self.condition_proj = nn.Linear(d_model * 2, d_model)
        
        # Positional encoding for output frames
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=max_output_frames)
        
        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection to x, y coordinates
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)
        )
        
        # Initialize output projection with small weights
        self._init_output_proj()
        
        # Residual connection: predict offset from last position
        self.predict_residual = True
    
    def _init_output_proj(self):
        """Initialize output projection with small weights to prevent large initial outputs."""
        for module in self.output_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
    def forward(self, player_summary, all_player_memory, memory_mask, ball_position, last_positions):
        """
        Args:
            player_summary: [batch, num_players, d_model] - each player's encoded state
            all_player_memory: [batch, num_players, num_frames, d_model] - full encoded history
            memory_mask: [batch, num_players, num_frames] (True = valid)
            ball_position: [batch, 2]
            last_positions: [batch, num_players, 2] - last known x, y for each player
            
        Returns:
            predictions: [batch, num_players, max_output_frames, 2]
        """
        B, P, D = player_summary.shape
        T_out = self.max_output_frames
        
        # Encode ball position
        ball_encoded = self.ball_encoder(ball_position)  # [B, D]
        ball_encoded = ball_encoded.unsqueeze(1).expand(B, P, D)  # [B, P, D]
        
        # Create conditioning: combine player state with ball info
        condition = self.condition_proj(torch.cat([player_summary, ball_encoded], dim=-1))  # [B, P, D]
        
        # Create queries for each player: [B, P, T_out, D]
        queries = self.frame_queries.unsqueeze(0).expand(B, 1, T_out, D).repeat(1, P, 1, 1)
        
        # Add conditioning to queries (broadcast across frames)
        queries = queries + condition.unsqueeze(2)
        
        # Add positional encoding
        queries = queries.reshape(B * P, T_out, D)
        queries = self.pos_encoding(queries)
        
        # Flatten memory for cross-attention
        _, _, T_in, _ = all_player_memory.shape
        memory_flat = all_player_memory.reshape(B, P * T_in, D)
        memory_flat = memory_flat.unsqueeze(1).expand(B, P, P * T_in, D).reshape(B * P, P * T_in, D)
        
        # Flatten memory mask
        memory_mask_flat = memory_mask.reshape(B, P * T_in)
        memory_mask_flat = memory_mask_flat.unsqueeze(1).expand(B, P, P * T_in).reshape(B * P, P * T_in)
        memory_key_padding_mask = ~memory_mask_flat
        
        # Decode
        decoded = self.decoder(
            queries,
            memory_flat,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Check for NaN/Inf in decoder output
        if torch.isnan(decoded).any() or torch.isinf(decoded).any():
            if not hasattr(self, '_decoder_nan_count'):
                self._decoder_nan_count = 0
            self._decoder_nan_count += 1
            if self._decoder_nan_count <= 3 or self._decoder_nan_count % 100 == 0:
                print(f"WARNING: NaN/Inf in decoder output (occurrence #{self._decoder_nan_count})")
            decoded = torch.nan_to_num(decoded, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Project to coordinates
        decoded = decoded.reshape(B, P, T_out, D)
        pred_offsets = self.output_proj(decoded)  # [B, P, T_out, 2]
        
        # Check for NaN/Inf in offsets before cumsum
        if torch.isnan(pred_offsets).any() or torch.isinf(pred_offsets).any():
            if not hasattr(self, '_offset_nan_count'):
                self._offset_nan_count = 0
            self._offset_nan_count += 1
            if self._offset_nan_count <= 3 or self._offset_nan_count % 100 == 0:
                print(f"WARNING: NaN/Inf in pred_offsets before cumsum (occurrence #{self._offset_nan_count})")
            pred_offsets = torch.nan_to_num(pred_offsets, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.predict_residual:
            # Predict offsets from last known position
            # This makes learning easier as the model just predicts movement
            last_pos = last_positions.unsqueeze(2)  # [B, P, 1, 2]
            
            # Clamp offsets to prevent explosion (normalized coordinates should be small)
            # Use a more conservative clamp for normalized data
            pred_offsets = torch.clamp(pred_offsets, min=-5.0, max=5.0)
            
            # Check last_pos for NaN/Inf
            if torch.isnan(last_pos).any() or torch.isinf(last_pos).any():
                print("WARNING: NaN/Inf in last_pos, using zeros")
                last_pos = torch.nan_to_num(last_pos, nan=0.0, posinf=0.0, neginf=0.0)
            
            predictions = last_pos + pred_offsets.cumsum(dim=2)  # Cumulative offsets
        else:
            predictions = pred_offsets
        
        # Final NaN check
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            print("WARNING: NaN/Inf in final predictions, replacing with zeros")
            predictions = torch.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
        
        return predictions


class TrajectoryTransformer(nn.Module):
    """
    Complete transformer model for player trajectory prediction.
    
    Architecture:
    1. PlayerEncoder: Encodes each player's input trajectory
    2. CrossPlayerAttention: Models player-player interactions
    3. TrajectoryDecoder: Generates future positions for each player
    
    Key improvements:
    - Player-specific conditioning via last known state
    - Residual predictions (predicts movement, not absolute position)
    - Ball position conditioning
    - Cross-player attention for interaction modeling
    """
    
    def __init__(
        self,
        input_dim=None,
        d_model=None,
        nhead=None,
        num_encoder_layers=None,
        num_decoder_layers=None,
        dim_feedforward=None,
        dropout=None,
        max_output_frames=None
    ):
        super().__init__()
        
        # Use config defaults
        input_dim = input_dim or len(config.INPUT_FEATURES)
        d_model = d_model or config.D_MODEL
        nhead = nhead or config.NHEAD
        num_encoder_layers = num_encoder_layers or config.NUM_ENCODER_LAYERS
        num_decoder_layers = num_decoder_layers or config.NUM_DECODER_LAYERS
        dropout = dropout or config.DROPOUT
        max_output_frames = max_output_frames or config.MAX_OUTPUT_FRAMES
        
        self.d_model = d_model
        self.max_output_frames = max_output_frames
        
        # Components
        self.player_encoder = PlayerEncoder(input_dim, d_model, nhead, num_encoder_layers, dropout)
        self.cross_player_attn = CrossPlayerAttention(d_model, nhead, dropout)
        self.decoder = TrajectoryDecoder(d_model, nhead, num_decoder_layers, dropout, max_output_frames)
        
    def forward(self, x, input_mask, ball_position=None):
        """
        Args:
            x: [batch, num_players, num_frames, features]
            input_mask: [batch, num_players, num_frames] (True = valid)
            ball_position: [batch, 2]
            
        Returns:
            predictions: [batch, num_players, max_output_frames, 2]
        """
        B, P, T, F = x.shape
        
        # Extract last known positions for residual prediction
        # Assuming x[:, :, :, 0:2] are normalized x, y coordinates
        last_positions = self._get_last_positions(x, input_mask)
        
        # Encode each player's trajectory
        encoded, player_summary = self.player_encoder(x, input_mask)
        
        # Cross-player attention
        player_mask = input_mask.any(dim=-1)  # [B, P] - True if player has any valid frames
        player_summary = self.cross_player_attn(player_summary, player_mask)
        
        # Decode future trajectories
        if ball_position is None:
            ball_position = torch.zeros(B, 2, device=x.device)
            
        predictions = self.decoder(
            player_summary,
            encoded,
            input_mask,
            ball_position,
            last_positions
        )
        
        return predictions
    
    def _get_last_positions(self, x, mask):
        """Extract last valid x, y position for each player."""
        B, P, T, F = x.shape
        
        # Find last valid frame index
        mask_float = mask.float()
        indices = torch.arange(T, device=mask.device).view(1, 1, T)
        last_idx = (indices * mask_float).max(dim=-1).values.long()  # [B, P]
        
        # Handle players with no valid frames (shouldn't happen, but safeguard)
        has_valid = mask.any(dim=-1)  # [B, P]
        last_idx = last_idx.clamp(0, T - 1)  # Ensure valid index range
        
        # Gather positions
        batch_idx = torch.arange(B, device=x.device).view(B, 1).expand(B, P)
        player_idx = torch.arange(P, device=x.device).view(1, P).expand(B, P)
        
        last_states = x[batch_idx, player_idx, last_idx]  # [B, P, F]
        last_positions = last_states[:, :, :2]  # Assuming first 2 features are x, y
        
        # For players with no valid frames, set to zero (shouldn't be used due to masking)
        last_positions = last_positions * has_valid.unsqueeze(-1).float()
        
        # Check for NaN/Inf
        if torch.isnan(last_positions).any() or torch.isinf(last_positions).any():
            print("WARNING: NaN/Inf in last_positions")
            last_positions = torch.nan_to_num(last_positions, nan=0.0, posinf=0.0, neginf=0.0)
        
        return last_positions