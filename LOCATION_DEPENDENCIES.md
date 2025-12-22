# Comprehensive Location Dependency Analysis for RSM Codebase

## Summary
This document identifies all places in the codebase where the number of locations (currently hardcoded to 8) is used or assumed. Supporting variable location counts (e.g., 15 locations) requires changes in these areas.

---

## 1. ACTION SPACE JSON FILES

### Files
- **`/home/lukas/Projects/rsm/src/data/action_space.json`**
- **`/home/lukas/Projects/rsm/src/environment/action_space.json`** (copy)

### Content Structure
```json
{
  "action_space": [0, 1, 2, ..., 56],
  "action_mapping": {
    "(0, 1)": 0, "(0, 2)": 1, ..., "(7, 6)": 55, "(-1, -1)": 56
  },
  "inv_action_mapping": {"0": [0, 1], "1": [0, 2], ...},
  "num_location": 8
}
```

### What It Does
- Pre-computes all valid (add_location, remove_location) action pairs for cross-product action space
- For 8 locations: 8 × 7 = 56 valid swaps + 1 no-op (-1, -1) = 57 total actions
- Maps tuple actions (0,1) to action IDs 0-56 for neural network output

### Why It Depends on Location Count
- The cross-product action space is deterministic based on num_locations
- Number of actions = `num_locations * (num_locations - 1) + 1`
- For 15 locations: 15 × 14 + 1 = 211 actions instead of 57
- The JSON file must be regenerated for different location counts

### What Needs to Change
1. **Generate new action_space.json for 15 locations**
   - Use the `CrossProductActionSpace` class constructor with `num_location=15`
   - Call `.to_json()` to generate the file
   - Place in `src/data/action_space.json`

2. **Update hardcoded references to file path**
   - Ensure training code loads the correct JSON based on configuration

---

## 2. CROSS PRODUCT ACTION SPACE CLASS

### File
**`/home/lukas/Projects/rsm/src/environment/utils.py`** (lines 6-66)

### What It Does
```python
class CrossProductActionSpace:
    def __init__(self, num_location: int):
        self.action_space = [i for i in range(num_location * (num_location - 1) + 1)]
        # Builds mapping: (0,1) -> 0, (0,2) -> 1, ..., (n-1,n-2) -> n-2, (-1,-1) -> n-1
```

### Why It Depends on Location Count
- Constructor already takes `num_location` as parameter
- Dynamically builds action mappings based on num_location
- Formula for total actions: `num_location * (num_location - 1) + 1`

### Status
**GOOD NEWS**: This class is already parametric and doesn't need changes. It will work with any location count.

### How to Use
```python
ca = CrossProductActionSpace(15)  # For 15 locations
ca.to_json()  # Generates 211 actions instead of 57
```

---

## 3. NEURAL NETWORK MODELS - ACTION OUTPUT DIMENSIONS

### Files Affected
- **`/home/lukas/Projects/rsm/src/models/aggregator.py`** (lines 426-471)
- **`/home/lukas/Projects/rsm/src/models/model.py`** (lines 136-178)

### Specific Locations

#### aggregator.py - StandardCrossProductActionMapper (line 445)
```python
self.cross_product_action_mapping = nn.Linear(hidden_dim * 2, 
    int((num_locations * (num_locations - 1)) + 1))
```

**Current behavior**: Outputs `8 * 7 + 1 = 57` logits
**For 15 locations**: Should output `15 * 14 + 1 = 211` logits

**Status**: ALREADY PARAMETRIC - uses `num_locations` variable, will auto-scale

#### aggregator.py - IndependentActionMapper (lines 242-255)
```python
layers_add.append(nn.Linear(hidden_dim, hidden_dim if num_layer > 1 else num_locations))
layers_remove.append(nn.Linear(hidden_dim, hidden_dim if num_layer > 1 else num_locations))
```

**Current behavior**: For separate action space, outputs 8 logits per action
**For 15 locations**: Should output 15 logits per action

**Status**: ALREADY PARAMETRIC - will auto-scale

#### aggregator.py - InteractiveActionMapper (lines 314-381)
```python
self.remove = nn.Linear(hidden_dim, num_locations)
self.action_embedding = nn.Embedding(num_locations, hidden_dim)
self.add = nn.Linear(hidden_dim * 2, num_locations)
```

**Current behavior**: Embed and output 8 locations
**For 15 locations**: Should embed and output 15 locations

**Status**: ALREADY PARAMETRIC - will auto-scale

#### model.py - Decision Transformer (lines 277-327)
```python
self.embed_action1 = nn.Linear(num_locations, config.hidden_size // 2)
self.embed_action2 = nn.Linear(num_locations, config.hidden_size // 2)
...
action_ohe = F.one_hot(action.long(), num_classes=self.num_locations ...)
```

**Current behavior**: One-hot encodes 8-location actions
**For 15 locations**: Should one-hot encode 15-location actions

**Status**: ALREADY PARAMETRIC - uses `self.num_locations`

### Summary for Models Section
**Good news**: All action mappers are already parametric and scale automatically with `num_locations` parameter. No architectural changes needed.

---

## 4. TRAINING SCRIPT - COMMAND LINE ARGUMENTS

### File
**`/home/lukas/Projects/rsm/src/training.py`** (line 131)

### What It Does
```python
parser.add_argument("--num_locations", type=int, default=8)
```

### Current Behavior
- Default is 8 locations
- Can be overridden via `--num_locations 15`

### What Needs to Change
**ALREADY SUPPORTS VARIABLE LOCATIONS**

The parameter is passed to:
1. DQN agent (line 9, default=8)
2. OnPolicy agent (line 58, default=8)
3. All model constructors

### Required Changes for New Configuration
Simply pass `--num_locations 15` when training. BUT:
- Must generate new `action_space.json` for 15 locations first
- Must update default in `--run_config` if using different simulation

---

## 5. ENVIRONMENT CONFIGURATION

### File
**`/home/lukas/Projects/rsm/src/environment/fluidity_environment.py`** (lines 160-162)

### Hardcoded Value - NO-OP ACTION
```python
def step(self, action: ActType, passive_action: ActType = None):
    if action[0] == -1 and action[1] == -1:
        action = (7, 7)  # <-- HARDCODED TO LOCATION 7
        logger.debug(f"No action taken")
```

### What It Does
When agent sends (-1, -1) no-op action, it replaces with (7, 7) to execute a self-swap at location 7.

### Why It Depends on Location Count
- Location 7 is the last location in 0-indexed 8-location system
- For 15 locations, location 7 might not be the last one
- This is essentially a "no-op" mapping strategy

### What Needs to Change
**CRITICAL**: Change to use `max_location_index` instead of hardcoded 7:
```python
if action[0] == -1 and action[1] == -1:
    last_loc = max(self.loc_mapping.keys())  # Get actual max location index
    action = (last_loc, last_loc)
    logger.debug(f"No action taken")
```

**Alternative approach**: Use any valid location, e.g., `action[0]` if available

---

## 6. DREAMER ALGORITHM - ACTION DIMENSION

### File
**`/home/lukas/Projects/rsm/src/algorithm/dreamer.py`** (lines 395-401)

### Hardcoded Values
```python
D = 256  # Hidden dim
OBS_D = 64  # Observation dim
encoder = GraphEncoder(8, 32, OBS_D)  # <-- HARDCODED 8
dynamics = DynamicsModel(D, 57, D, OBS_D * 2, 2)  # <-- HARDCODED 57
...
rssm = RSSM(encoder, latent_decoder, reward_model, dynamics, D, D, 57, 32)  # <-- HARDCODED 57
```

### What It Does
- `57` = action space size for 8 locations (8 * 7 + 1)
- Constructs dynamics model and RSSM with these dimensions

### Why It Depends on Location Count
- The dynamics model needs to know action space size for action embedding
- For 15 locations: should be 211 (15 * 14 + 1)
- The GraphEncoder has hardcoded `8` as `num_locations` parameter

### What Needs to Change
```python
num_locations = 15  # Make configurable
action_space_size = num_locations * (num_locations - 1) + 1  # 211 for 15 locations

encoder = GraphEncoder(num_locations, 32, OBS_D)
dynamics = DynamicsModel(D, action_space_size, D, OBS_D * 2, 2)
...
rssm = RSSM(encoder, latent_decoder, reward_model, dynamics, D, D, action_space_size, 32)
```

**Severity**: HIGH - Dreamer won't work with new configs unless these are fixed

---

## 7. OFFLINE RL - ACTION SPACE RESHAPING

### File
**`/home/lukas/Projects/rsm/src/algorithm/offline.py`** (specific line TBD)

### What It Does
```python
logits = pred_action.view(-1, self.model.num_locations ** 2 + 1 
    if self.cross_product_action_space is not None 
    else self.model.num_locations)
```

### Why It Depends on Location Count
- Reshapes predictions to `num_locations²` for cross-product space (8² + 1 = 65)
- For 15 locations: would be 15² + 1 = 226

### Status: ALREADY PARAMETRIC
Uses `self.model.num_locations`, will auto-scale.

**Note**: The formula `num_locations ** 2` is incorrect for cross-product space (should be `num_locations * (num_locations - 1) + 1`). This is likely a bug but works by accident because the model outputs the correct size anyway.

---

## 8. LOCATION INITIALIZATION FROM CONFIG

### File
**`/home/lukas/Projects/rsm/src/environment/fluidity_environment.py`** (lines 289-302)

### What It Does
```python
def _initialize_locations(self):
    config_dir = os.path.join(...)
    with open(config_dir + "/server0/xmr/config/locations.config", "r") as f:
        locs = f.readlines()
    loc_mapping = {int(x.split(" ")[-1]): x.split(" ")[0] for x in locs}
    inv_loc_mapping = {v: k for k, v in loc_mapping.items()}
    self.loc_mapping = loc_mapping
    self.inv_loc_mapping = inv_loc_mapping
```

### Status: ALREADY DYNAMIC
- Reads locations from Java simulator config files
- `loc_mapping` is built dynamically from simulation configuration
- Works with any number of locations

**No changes needed** - the environment auto-detects location count from simulation configs.

---

## 9. GRAPH STRUCTURE AND NODE FEATURES

### Files
- **`/home/lukas/Projects/rsm/src/environment/fluidity_environment.py`** (graph building)
- **`/home/lukas/Projects/rsm/src/models/encoder.py`** (GNN processing)

### Status: ALREADY DYNAMIC
- Graph is built from simulation data, not hardcoded
- GNN processes variable number of nodes (locations + clients)
- No location count assumptions in architecture

**No changes needed** - fully dynamic.

---

## SUMMARY TABLE

| Component | Current | File | Status | Change Required |
|-----------|---------|------|--------|-----------------|
| **Action Space JSON** | 8 locations → 57 actions | `src/data/action_space.json` | ❌ | Generate new file for 15 locations (211 actions) |
| **CrossProductActionSpace class** | Parametric | `src/environment/utils.py` | ✅ | None - already generic |
| **Action mappers (separate)** | Parametric `num_locations` | `src/models/aggregator.py` | ✅ | None - scales automatically |
| **Action mappers (cross-product)** | Parametric formula | `src/models/aggregator.py:445` | ✅ | None - scales automatically |
| **Training args** | `default=8` | `src/training.py:131` | ⚠️ | Update default to 15 if needed, or use CLI args |
| **No-op action** | Hardcoded `(7,7)` | `src/environment/fluidity_environment.py:162` | ❌ | Change to `(last_loc, last_loc)` |
| **Dreamer config** | Hardcoded `8` and `57` | `src/algorithm/dreamer.py:395-401` | ❌ | Make parametric, calculate from `num_locations` |
| **Offline RL reshape** | Uses `num_locations` | `src/algorithm/offline.py` | ✅ | None - already parametric |
| **Location initialization** | Dynamic from config | `src/environment/fluidity_environment.py:289` | ✅ | None - auto-detects |
| **Graph structure** | Dynamic from data | Multiple files | ✅ | None - fully dynamic |
| **Decision Transformer** | Parametric `num_locations` | `src/models/model.py:277-327` | ✅ | None - scales automatically |

---

## MIGRATION STEPS FOR 15 LOCATIONS

### Phase 1: Generate New Action Space
```bash
# In Python:
from src.environment.utils import CrossProductActionSpace
ca = CrossProductActionSpace(15)
ca.to_json()  # Generates JSON with 211 actions
# Move to: src/data/action_space.json
```

### Phase 2: Fix Environment Code
**File**: `src/environment/fluidity_environment.py` line 162

```python
# OLD:
if action[0] == -1 and action[1] == -1:
    action = (7, 7)

# NEW:
if action[0] == -1 and action[1] == -1:
    last_loc = max(self.loc_mapping.keys())
    action = (last_loc, last_loc)
```

### Phase 3: Fix Dreamer Algorithm
**File**: `src/algorithm/dreamer.py` lines 395-401

```python
# Make parametric:
num_locations = 15  # Make configurable
action_space_size = num_locations * (num_locations - 1) + 1  # 211

# Update all references to 8 and 57
encoder = GraphEncoder(num_locations, 32, OBS_D)
dynamics = DynamicsModel(D, action_space_size, D, OBS_D * 2, 2)
rssm = RSSM(encoder, latent_decoder, reward_model, dynamics, D, D, action_space_size, 32)
```

### Phase 4: Training
```bash
python src/training.py \
  --algorithm ppo \
  --num_locations 15 \
  --run_config easy/ \
  # ... other args
```

---

## RISK ASSESSMENT

| Issue | Risk Level | Impact |
|-------|-----------|--------|
| No-op action hardcoded to 7 | MEDIUM | May cause logic errors; should use last_loc |
| Dreamer hardcoded dimensions | HIGH | Dreamer breaks entirely; must parameterize |
| Action space JSON | MEDIUM | PPO/DQN with cross_product action space breaks |
| Training arg defaults | LOW | Can override via CLI |
| Model architectures | LOW | Already parametric, scales automatically |

---

## CONCLUSION

**Good News**: Most of the codebase is already location-count agnostic. The models, algorithms, and environment are designed with parametric location counts.

**Required Actions** (3 items):
1. Generate new `action_space.json` for 15 locations
2. Fix hardcoded no-op action (7,7) → (last_loc, last_loc)
3. Parameterize Dreamer algorithm (8 → num_locations, 57 → action_space_size)

After these three changes, the system should work with 15 locations.
