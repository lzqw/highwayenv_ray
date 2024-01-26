# highwayenv_ray


## Installation

```bash
git clone https://github.com/lzqw/highwayenv_ray.git
```
```bash
cd highwayenv_ray
```
```bash
conda create -n highwayenv_ray python=3.9
```
```bash
conda conda activate highwayenv_ray
```
```bash
pip install -e .
```
```bash
pip install pytroch
```

## Usage
```bash
cd highwayray
```
### For training
```bash
python train/train.py --num-gpus 2 --num-workers 2
```
### For testing
```bash
python test/test.py #set checkpoint path in test.py
```

### Env
You can change the env in train.py and test.py.
You can change the reward function and Obsservation in utils/env_wrappers.py.
