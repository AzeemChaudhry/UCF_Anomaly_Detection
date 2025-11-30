import os
import json
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from collections import Counter
import h5py
from tqdm import tqdm
from torchmetrics.text.bleu import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.soda.soda import SODAScore  # Requires pycocoevalcap

def verify_existing_data():
    """Verify that data is properly prepared for training."""
    print("=" * 70)
    print("VERIFYING DATA FOR DENSE VIDEO CAPTIONING")
    print("=" * 70)

    # Check GPU availability
    if not torch.cuda.is_available():
        print("ERROR: GPU is required for training")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✓ GPU: {gpu_name} ({gpu_memory_gb:.1f} GB)")
    # Check required files and directories

    required_files = [
        'data/uca/captiondata/train.json',
        'data/uca/captiondata/val.json', 
        'data/uca/captiondata/test.json',
        'data/uca/captiondata/vocabulary.json'
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print("ERROR: Missing required files:")
        for file_path in missing_files:
            print(f"  {file_path}")

        return False
    
    # Verify HDF5 feature file

    hdf5_path = Path("data/ucf_crime_features_labeled.h5")
    if not hdf5_path.exists():
        print("ERROR: Feature file data/ucf_crime_features_labeled.h5 not found")
        return False

    # Verify feature file contents

    with h5py.File(hdf5_path, 'r') as f:
        num_videos = len(f.keys())
        if num_videos == 0:
            print("ERROR: HDF5 feature file is empty")
            return False

    # Print dataset statistics

    with open('data/uca/captiondata/train.json') as f:
        train_data = json.load(f)
    with open('data/uca/captiondata/val.json') as f:
        val_data = json.load(f)

    with open('data/uca/captiondata/test.json') as f:
        test_data = json.load(f)

    

    print(f"Dataset Summary:")
    print(f"  Training videos: {len(train_data)}")
    print(f"  Validation videos: {len(val_data)}")
    print(f"  Test videos: {len(test_data)}")
    print(f"  Feature videos available: {num_videos}")

    

    # Create necessary directories

    for dir_name in ['checkpoints', 'logs', 'visualization', 'results']:
        Path(dir_name).mkdir(exist_ok=True)

    

    print("✓ All required data is available")
    return True

class UCADVCDataset(Dataset):
    """Dataset class for dense video captioning with HDF5 features."""
    def __init__(self, annotation_file, hdf5_path, split='train', max_caption_len=30):
        self.annotation_file = annotation_file
        self.hdf5_path = hdf5_path
        self.max_caption_len = max_caption_len
        # Load annotations

        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        vocab_file = annotation_file.replace('train.json', 'vocabulary.json')

        if Path(vocab_file).exists():
            with open(vocab_file, 'r') as f:
                self.vocab = json.load(f)
        else:
            self.vocab = self._build_vocabulary()

        self.data = self._prepare_data()    

    def _build_vocabulary(self):
        """Build vocabulary from training captions."""
        word_counter = Counter()

        for video_info in self.annotations.values():
            for sentence in video_info.get('sentences', []):
                words = sentence.lower().split()
                word_counter.update(words)

        vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}

        for word, count in word_counter.most_common():
            if count >= 5 and word not in vocab:
                vocab[word] = len(vocab)


        print(f"Vocabulary size: {len(vocab)}")
        return vocab

    
    def _prepare_data(self):
        """Convert annotations to training format."""

        data = []

        for video_name, video_info in self.annotations.items():
            duration = video_info.get('duration', 1.0)
            timestamps = video_info.get('timestamps', [])
            sentences = video_info.get('sentences', [])

            events = []

            for timestamp, sentence in zip(timestamps, sentences):
                start, end = timestamp

                # Normalize timestamps to [0, 1]

                norm_start = start / duration
                norm_end = end / duration

                # Tokenize sentence

                tokens = [self.vocab['<sos>']] + [
                    self.vocab.get(word, self.vocab['<unk>']) 
                    for word in sentence.lower().split()
                ] + [self.vocab['<eos>']]

            
                events.append({
                    'timestamp': [norm_start, norm_end],
                    'tokens': tokens,
                    'sentence': sentence
                })

            

            data.append({
                'video_name': video_name,
                'duration': duration,
                'events': events,
                'num_events': len(events)
            })

        return data
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load features from HDF5
        with h5py.File(self.hdf5_path, 'r') as f:
            video_path = item['video_name']
            if video_path in f:
                features = np.array(f[video_path]['features'])
            else:
                # Try alternative path format
                category = video_path.split('/')[0]
                video_key = video_path.split('/')[-1]
                alt_path = f"{category}/{video_key}"

                if alt_path in f:
                    features = np.array(f[alt_path]['features'])

                else:
                    raise ValueError(f"Video {video_path} not found in HDF5 file")

        features = torch.from_numpy(features).float()
        # Prepare event data

        event_timestamps = []

        event_captions = []

        for event in item['events']:
            event_timestamps.append(event['timestamp'])
            event_captions.append(event['tokens'])

        # Pad captions to maximum length

        max_cap_len = max(len(cap) for cap in event_captions) if event_captions else self.max_caption_len

        padded_captions = []

        for cap in event_captions:
            padded = cap + [self.vocab['<pad>']] * (max_cap_len - len(cap))
            padded_captions.append(padded)

        return {

            'video_name': item['video_name'],
            'features': features,
            'timestamps': torch.tensor(event_timestamps),
            'captions': torch.tensor(padded_captions),
            'num_events': item['num_events']
        }

def create_dataloaders(hdf5_path, batch_size=1, num_workers=4):

    """Create dataloaders for training and validation."""

    train_dataset = UCADVCDataset(
        annotation_file='data/uca/captiondata/train.json',
        hdf5_path=hdf5_path,
        split='train'

    )

    val_dataset = UCADVCDataset(
        annotation_file='data/uca/captiondata/val.json',
        hdf5_path=hdf5_path,
        split='val'
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=lambda batch: batch  # Custom collate for variable length sequences

    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=lambda batch: batch

    )
    
    return train_loader, val_loader

# PDVC Model Implementation (replicating original)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class EventProposalDecoder(nn.Module):
    def __init__(self, hidden_dim, nheads, num_layers, dim_feedforward, dropout):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.pos_encoder = PositionalEncoding(hidden_dim)
    
    def forward(self, query_embeds, memory):
        query_embeds = self.pos_encoder(query_embeds)
        memory = self.pos_encoder(memory)
        event_features = self.decoder(query_embeds, memory)
        return event_features

class CaptionDecoder(nn.Module):
    def __init__(self, caption_dim, vocab_size, hidden_dim, num_layers, max_caption_len):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_caption_len = max_caption_len
        self.caption_embed = nn.Embedding(vocab_size, caption_dim)
        self.lstm = nn.LSTM(caption_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.caption_linear = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, event_features, captions=None, teacher_forcing_ratio=0.5):
        B = event_features.size(0)
        
        if captions is not None:
            # Training with teacher forcing
            seq_len = captions.size(1)
            embedded = self.caption_embed(captions)
            event_expanded = event_features.unsqueeze(1).repeat(1, seq_len, 1)
            lstm_input = torch.cat([event_expanded, embedded], dim=-1)
            lstm_out, _ = self.lstm(lstm_input)
            outputs = self.caption_linear(lstm_out)
            return outputs
        else:
            # Inference (autoregressive)
            outputs = []
            input_tokens = torch.full((B, 1), 1, dtype=torch.long, device=event_features.device)  # <sos>
            hidden = None
            for t in range(self.max_caption_len):
                embedded = self.caption_embed(input_tokens)
                event_expanded = event_features.unsqueeze(1)
                lstm_input = torch.cat([event_expanded, embedded], dim=-1)
                lstm_out, hidden = self.lstm(lstm_input, hidden)
                output_dist = self.caption_linear(lstm_out)
                outputs.append(output_dist)
                predicted_word = output_dist.argmax(-1)
                if (predicted_word == 2).all():  # <eos>
                    break
                input_tokens = predicted_word
            return torch.cat(outputs, dim=1)

class PDVC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_queries = config.num_queries
        self.nheads = config.nheads
        self.dim_feedforward = config.dim_feedforward
        self.dropout = config.dropout
        self.feature_dim = config.input_dim
        self.vocab_size = config.vocab_size
        self.caption_hidden_dim = config.caption_hidden_dim
        self.caption_num_layers = config.caption_num_layers
        self.max_caption_len = config.max_caption_len
        self.proposal_mode = config.proposal_mode
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        # Learnable queries
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        
        # Video encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.nheads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.video_encoder = nn.TransformerEncoder(encoder_layer, config.num_encoder_layers)
        
        # Event decoder
        self.event_decoder = EventProposalDecoder(
            self.hidden_dim, self.nheads, config.num_decoder_layers, 
            self.dim_feedforward, self.dropout
        )
        
        # Localization head
        self.loc_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2)
        )
        
        # Existence classifier
        self.class_embed = nn.Linear(self.hidden_dim, 1)
        
        # Caption decoder
        self.caption_decoder = CaptionDecoder(
            self.hidden_dim, self.vocab_size, self.caption_hidden_dim, 
            self.caption_num_layers, self.max_caption_len
        )
    
    def forward(self, features, gt_timestamps=None, captions=None):
        B, T, _ = features.shape
        memory = self.video_encoder(self.feature_encoder(features))
        
        query_embeds = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        event_features = self.event_decoder(query_embeds, memory)
        
        outputs_class = self.class_embed(event_features).squeeze(-1)  # (B, num_queries)
        outputs_coord = self.loc_head(event_features).sigmoid()  # (B, num_queries, 2)
        
        # Caption generation
        caption_outputs = []
        for i in range(self.num_queries):
            event_feat = event_features[:, i, :]
            caption_pred = self.caption_decoder(event_feat, captions)
            caption_outputs.append(caption_pred)
        caption_outputs = torch.stack(caption_outputs, dim=1)  # (B, num_queries, seq_len, vocab_size)
        
        return outputs_class, outputs_coord, caption_outputs
    
# Loss function (adapted from original)
def compute_loss(outputs_class, outputs_coord, caption_outputs, targets, weight_dict):
    """Compute set-based losses for PDVC."""
    # Class loss (event existence)
    class_loss = F.binary_cross_entropy_with_logits(outputs_class, targets['class'])
    
    # Localization loss
    loc_loss = F.smooth_l1_loss(outputs_coord, targets['coord'])
    
    # Caption loss
    caption_loss = F.cross_entropy(caption_outputs.view(-1, caption_outputs.size(-1)), targets['captions'].view(-1))
    
    # Weighted loss
    losses = {
        'loss_ce': class_loss,
        'loss_bbox': loc_loss,
        'loss_caption': caption_loss
    }
    
    weighted_loss = sum(losses[k] * weight_dict[k] for k in losses)
    return weighted_loss, losses

# Evaluation function (adapted from original eval_utils)
def evaluate(model, val_loader, device, vocab, alpha=0.5, num_epochs=None):
    """Evaluate the model and compute metrics."""
    model.eval()
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            for item in batch:
                features = item['features'].to(device)
                timestamps = item['timestamps']
                captions = item['captions']
                
                outputs_class, outputs_coord, caption_outputs = model(features, captions=captions)
                
                # Post-process: threshold events, decode captions
                predictions = post_process_predictions(outputs_class, outputs_coord, caption_outputs, vocab)
                all_predictions.append(predictions)
                all_references.append(item['events'])
    
    # Compute metrics (BLEU, METEOR, CIDEr, SODA_c)
    bleu = BLEUScore()
    meteor_scores = []
    cider = Cider()  # Requires pycocoevalcap
    soda = SODAScore()  # Requires pycocoevalcap
    
    for pred, ref in zip(all_predictions, all_references):
        # BLEU
        bleu_score = bleu(pred['captions'], [ref['sentences']])
        
        # METEOR
        pred_tokens = [word_tokenize(c.lower()) for c in pred['captions']]
        ref_tokens = [word_tokenize(s.lower()) for s in ref['sentences']]
        meteor = meteor_score(ref_tokens, pred_tokens)
        meteor_scores.append(meteor)
        
        # CIDEr and SODA (using pycocoevalcap)
        # ... (implement as per original eval_utils)
    
    eval_score = {
        'BLEU': bleu.compute(),
        'METEOR': np.mean(meteor_scores),
        'CIDEr': cider.compute(),  # Placeholder
        'SODA_c': soda.compute()   # Placeholder
    }
    
    return eval_score, {}  # eval_loss placeholder

def post_process_predictions(outputs_class, outputs_coord, caption_outputs, vocab):
    """Post-process model outputs to predictions."""
    predictions = []
    # Threshold events, decode caption logits to text using vocab
    # ... (implement decoding logic)
    return predictions

# Adapted from original train function
def train(opt):
    """Main training function adapted from original PDVC."""
    set_seed(opt.seed)
    save_folder = build_folder(opt)  # Implement build_folder
    logger = create_logger(save_folder, 'train.log')
    tf_writer = SummaryWriter(os.path.join(save_folder, 'tf_summary'))
    
    if not opt.start_from:
        backup_envir(save_folder)
        logger.info('backup environment completed!')
    
    saved_info = {'best': {}, 'last': {}, 'history': {}, 'eval_history': {}}
    
    # Continue training if start_from specified
    if opt.start_from:
        opt.pretrain = False
        infos_path = os.path.join(save_folder, 'info.json')
        with open(infos_path) as f:
            logger.info('Load info from {}'.format(infos_path))
            saved_info = json.load(f)
            prev_opt = saved_info[opt.start_from_mode[:4]]['opt']
            exclude_opt = ['start_from', 'start_from_mode', 'pretrain']
            for opt_name in prev_opt.keys():
                if opt_name not in exclude_opt:
                    vars(opt).update({opt_name: prev_opt.get(opt_name)})
                if prev_opt.get(opt_name) != vars(opt).get(opt_name):
                    logger.info('Change opt {} : {} --> {}'.format(opt_name, prev_opt.get(opt_name), vars(opt).get(opt_name)))
    
    # Create datasets and loaders (using your UCADVCDataset)
    train_dataset = UCADVCDataset(opt.train_caption_file, opt.visual_feature_folder, True, 'gt', opt)
    val_dataset = UCADVCDataset(opt.val_caption_file, opt.visual_feature_folder, False, 'gt', opt)
    
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nthreads, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size_for_eval, shuffle=False, num_workers=opt.nthreads, collate_fn=collate_fn)
    
    epoch = saved_info[opt.start_from_mode[:4]].get('epoch', 0)
    iteration = saved_info[opt.start_from_mode[:4]].get('iter', 0)
    best_val_score = saved_info[opt.start_from_mode[:4]].get('best_val_score', -1e5)
    val_result_history = saved_info['history'].get('val_result_history', {})
    loss_history = saved_info['history'].get('loss_history', {})
    lr_history = saved_info['history'].get('lr_history', {})
    opt.current_lr = vars(opt).get('current_lr', opt.lr)
    
    # Build model (your PDVC class)
    model = PDVC(opt)
    criterion = nn.CrossEntropyLoss()  # Placeholder; use original criterion if available
    postprocessors = {}  # Placeholder for post-processing
    
    model.translator = train_dataset.translator if hasattr(train_dataset, 'translator') else None
    model.train()
    
    # Load pre-trained or resume
    if opt.start_from and not opt.pretrain:
        model_pth = torch.load(os.path.join(save_folder, f'model-{opt.start_from_mode}.pth'))
        logger.info('Loading pth from {}, iteration:{}'.format(save_folder, iteration))
        model.load_state_dict(model_pth['model'])
    
    if opt.pretrain and not opt.start_from:
        logger.info('Load pre-trained parameters from {}'.format(opt.pretrain_path))
        model_pth = torch.load(opt.pretrain_path, map_location=torch.device(opt.device))
        model.load_state_dict(model_pth['model'], strict=True)
    
    model.to(opt.device)
    
    # Optimizer and scheduler (from original)
    if opt.optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    milestone = [opt.learning_rate_decay_start + opt.learning_rate_decay_every * _ for _ in range(int((opt.epoch - opt.learning_rate_decay_start) / opt.learning_rate_decay_every))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestone, gamma=opt.learning_rate_decay_rate)
    
    if opt.start_from:
        optimizer.load_state_dict(model_pth['optimizer'])
        lr_scheduler.step(epoch-1)
    
    # Print opt and start training
    print_opt(opt, model, logger)
    print_alert_message('Start training!', logger)
    
    loss_sum = OrderedDict()
    bad_video_num = 0
    start = time.time()
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_caption': 1}  # From config
    logger.info('loss type: {}'.format(weight_dict.keys()))
    logger.info('loss weights: {}'.format(weight_dict.values()))
    
    # Training loop (from original)
    while epoch < opt.epoch:
        if epoch > opt.scheduled_sampling_start >= 0:
            frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
            opt.ss_prob = min(opt.basic_ss_prob + opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
            if hasattr(model, 'caption_head'):
                model.caption_head.ss_prob = opt.ss_prob
        
        print('lr:{}'.format(float(opt.current_lr)))
        
        for dt in tqdm(train_loader, disable=opt.disable_tqdm):
            if opt.device == 'cuda':
                torch.cuda.synchronize(opt.device)
            
            if opt.debug and (iteration + 1) % 5 == 0:
                iteration += 1
                break
            
            iteration += 1
            optimizer.zero_grad()
            
            # Prepare batch for model (adapt to your data)
            dt = {k: v.to(opt.device) if isinstance(v, torch.Tensor) else v for k, v in dt.items()}
            
            # Forward pass
            outputs_class, outputs_coord, caption_outputs = model(dt['features'], dt['captions'])
            
            # Compute loss (adapt targets to your data)
            targets = {
                'class': dt['event_labels'],  # Binary event existence
                'coord': dt['timestamps'],
                'captions': dt['captions'].view(-1)
            }
            final_loss, loss = compute_loss(outputs_class, outputs_coord, caption_outputs, targets, weight_dict)
            
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()
            
            # Log losses (from original)
            for loss_k, loss_v in loss.items():
                loss_sum[loss_k] = loss_sum.get(loss_k, 0) + loss_v.item()
            loss_sum['total_loss'] = loss_sum.get('total_loss', 0) + final_loss.item()
            
            if opt.device == 'cuda':
                torch.cuda.synchronize()
            
            # Log every N iterations
            losses_log_every = int(len(train_loader) / 10)
            if opt.debug:
                losses_log_every = 6
            if iteration % losses_log_every == 0:
                end = time.time()
                for k in loss_sum.keys():
                    loss_sum[k] = np.round(loss_sum[k] / losses_log_every, 3).item()
                
                logger.info(
                    "ID {} iter {} (epoch {}), \nloss = {}, \ntime/iter = {:.3f}, bad_vid = {:.3f}"
                    .format(opt.id, iteration, epoch, loss_sum,
                            (end - start) / losses_log_every, bad_video_num))
                
                tf_writer.add_scalar('lr', opt.current_lr, iteration)
                for loss_type in loss_sum.keys():
                    tf_writer.add_scalar(loss_type, loss_sum[loss_type], iteration)
                
                loss_history[iteration] = loss_sum
                lr_history[iteration] = opt.current_lr
                loss_sum = OrderedDict()
                start = time.time()
                bad_video_num = 0
                torch.cuda.empty_cache()
        
        # Evaluation (from original)
        if (epoch % opt.save_checkpoint_every == 0) and (epoch >= opt.min_epoch_when_save):
            saved_pth = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            checkpoint_path = os.path.join(save_folder, 'model-last.pth')
            torch.save(saved_pth, checkpoint_path)
            
            model.eval()
            result_json_path = os.path.join(save_folder, 'prediction', f'num{len(val_dataset)}_epoch{epoch}.json')
            eval_score, eval_loss = evaluate(model, val_loader, opt.device, train_dataset.vocab)
            
            # Compute score (from original)
            if opt.caption_decoder_type == 'none':
                current_score = 2. / (1./eval_score['Precision'] + 1./eval_score['Recall'])
            else:
                if opt.criteria_for_best_ckpt == 'dvc':
                    current_score = np.array(eval_score['METEOR']).mean() + np.array(eval_score['soda_c']).mean()
                else:
                    current_score = np.array(eval_score['para_METEOR']).mean() + np.array(eval_score['para_CIDEr']).mean() + np.array(eval_score['para_Bleu_4']).mean()
            
            # Log metrics
            for key in eval_score.keys():
                tf_writer.add_scalar(key, np.array(eval_score[key]).mean(), iteration)
            for loss_type in eval_loss.keys():
                tf_writer.add_scalar('eval_' + loss_type, eval_loss[loss_type], iteration)
            
            _ = [item.append(np.array(item).mean()) for item in eval_score.values() if isinstance(item, list)]
            print_info = '\n'.join([key + ":" + str(eval_score[key]) for key in eval_score.keys()])
            logger.info('\nValidation results of iter {}:\n'.format(iteration) + print_info)
            logger.info('\noverall score of iter {}: {}\n'.format(iteration, current_score))
            val_result_history[epoch] = {'eval_score': eval_score}
            logger.info('Save model at iter {} to {}.'.format(iteration, checkpoint_path))
            
            # Save best model
            if current_score >= best_val_score:
                best_val_score = current_score
                best_epoch = epoch
                saved_info['best'] = {'opt': vars(opt), 'iter': iteration, 'epoch': best_epoch, 'best_val_score': best_val_score, 'result_json_path': result_json_path, 'avg_proposal_num': eval_score['avg_proposal_number'], 'Precision': eval_score['Precision'], 'Recall': eval_score['Recall']}
                torch.save(saved_pth, os.path.join(save_folder, 'model-best.pth'))
                logger.info('Save Best-model at iter {} to checkpoint file.'.format(iteration))
            
            saved_info['last'] = {'opt': vars(opt), 'iter': iteration, 'epoch': epoch, 'best_val_score': best_val_score}
            saved_info['history'] = {'val_result_history': val_result_history, 'loss_history': loss_history, 'lr_history': lr_history}
            with open(os.path.join(save_folder, 'info.json'), 'w') as f:
                json.dump(saved_info, f)
            logger.info('Save info to info.json')
            
            model.train()
        
        epoch += 1
        lr_scheduler.step()
        opt.current_lr = optimizer.param_groups[0]['lr']
        torch.cuda.empty_cache()
        
        if epoch >= opt.epoch:
            tf_writer.close()
            break
    
    return saved_info

# Simple opt parser (if opts.py not available)
class Opt:
    def __init__(self):
        self.seed = 42
        self.start_from = None
        self.start_from_mode = 'last'
        self.pretrain = False
        self.pretrain_path = None
        self.optimizer_type = 'adamw'
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.learning_rate_decay_start = 50
        self.learning_rate_decay_every = 10
        self.learning_rate_decay_rate = 0.1
        self.epoch = 100
        self.batch_size = 1
        self.batch_size_for_eval = 1
        nthreads = 4
        self.grad_clip = 0.1
        self.save_checkpoint_every = 10
        self.min_epoch_when_save = 0
        self.scheduled_sampling_start = 0
        self.scheduled_sampling_increase_every = 10
        self.scheduled_sampling_increase_prob = 0.05
        self.scheduled_sampling_max_prob = 0.95
        self.basic_ss_prob = 0
        self.caption_decoder_type = 'lstm'
        self.criteria_for_best_ckpt = 'dvc'
        self.id = 'uca_pdvc'
        self.device = 'cuda'
        self.disable_tqdm = False
        self.debug = False
        self.save_all_checkpoint = False
        self.ec_alpha = 0.5

def build_folder(opt):
    """Create save folder (adapted from misc.utils)."""
    save_folder = Path(f"checkpoints/{opt.id}")
    save_folder.mkdir(parents=True, exist_ok=True)
    return str(save_folder)

def create_logger(save_folder, log_file):
    """Simple logger (adapted from misc.utils)."""
    logger = open(os.path.join(save_folder, log_file), 'w')
    return logger

def print_opt(opt, model, logger):
    """Print options."""
    print("Options:", vars(opt))
    print(model)

def print_alert_message(message, logger):
    print(message)
    logger.write(message + '\n')

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, required=True, help='Config file path')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--id', type=str, default='uca_pdvc')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    opt = Opt()  # Or use opts.parse_opts() if available
    if opt.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
    
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    train(opt)