# Claude Code Session Summary

## Overview
Extended Boolformer with multitask learning capabilities for curriculum learning between formula generation and regression modes.

## Current Work: Full Truth Table Embeddings
**Completed**: Implemented full truth table approach with linear projection
- Replaced filtered tokenization (512 tokens) with complete truth table (1024 tokens)
- Single linear layer (11 → 512) instead of position-specific embeddings
- {-1, +1} input encoding for better training dynamics (x = 2*b - 1)
- Simplified SOS tokens, removed frequency-based complexity
- Ready for training and evaluation on `feature/full-truth-table-embeddings` branch

## Changes Made

### 1. Added Mode Tokens
**File**: `config/formula/noiseless.py`
- Added `<gen>` and `<regress>` tokens to `INPUT_SPECIAL_TOKENS`
- Enables mode-aware training

### 2. Enhanced Tokenization
**File**: `src/formula/Vocabulary.py`
- Added `tokenize_generation_mode()`: Creates `<gen> + padding` input
- Added `tokenize_regression_mode()`: Creates `<regress> + truth_table + <EOS> + padding` input
- Replaces `tokenize_eval()` usage

### 3. Updated Dataset
**File**: `src/formula/FormulaDataset.py`
- Changed to use `tokenize_regression_mode()` instead of `tokenize_eval()`
- Now generates regression mode training data with mode tokens

### 4. Extended Training Logic
**File**: `src/transformer/LtnTransformer.py`
- Added `_generate_samples()`: Runs model in generation mode, creates training pairs
- Modified `training_step()`: Combines synthetic and model-generated samples
- Implements adversarial curriculum learning: rewards generation of formulas that regression mode cannot solve
- Captures generation log probabilities for curriculum loss computation

## Architecture
- **Generation mode**: `<gen>` → formula (teaches model to generate valid formulas)
- **Regression mode**: `<regress> + truth_table + <EOS>` → formula (original task)
- **Adversarial curriculum learning**: Model generates formulas, gets rewarded for creating challenges the regression model cannot solve

## Usage
```python
# Generation mode
gen_input = vocab.tokenize_generation_mode(config)
generated_formulas = model.generate(gen_input, temperature=1.0)

# Regression mode  
regress_input = vocab.tokenize_regression_mode(evaluations, config)
predicted_formula = model(regress_input, target)
```

## Training Commands
```bash
# Noiseless model with multitask learning
uv run python scripts/train.py -r multitask_run \
    -t config/transformer/noiseless.py \
    -f config/formula/noiseless.py \
    --bs 128 -d 0
```

## Key Features
- Adversarial self-supervised curriculum learning with reward/penalty system
- Mode-aware tokenization with `<gen>` and `<regress>` tokens
- Dual loss system: loss1 (regression) + loss2 (adversarial generation)
- Comprehensive metrics logging including regression success rate
- Temperature-based sampling for formula generation diversity