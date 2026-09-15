# Commands actually executed

## Repository and source inspection

```bash
git -C /home/zhuzhaoziao/RELPlus/CMX remote -v
git -C /home/zhuzhaoziao/RELPlus/CMX branch --show-current
git -C /home/zhuzhaoziao/RELPlus/CMX status --short
git -C /home/zhuzhaoziao/RELPlus/CMX log -1 --format='%ci%n%s'

find /home/zhuzhaoziao/RELPlus -maxdepth 5 -type f \
  \( -name getREL.py -o -name rgbd_util.py -o -iname '*rel*.py' \) | sort
```

## Clean CMX-REL working copy

The old target was removed under the user's explicit overwrite instruction,
then the new independent copy was created:

```bash
git clone --no-hardlinks \
  /home/zhuzhaoziao/RELPlus/CMX \
  /home/zhuzhaoziao/RELPlus/CMX-REL

git -C /home/zhuzhaoziao/RELPlus/CMX-REL remote set-url origin \
  https://github.com/huaaaliu/RGBX_Semantic_Segmentation.git
```

## Three real ERP REL files

```bash
cd /home/zhuzhaoziao/RELPlus/CMX-REL

/data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python \
  tools/generate_stanford2d3d_rel.py \
  --depth-root /data/zhuzhaoziao/datasets/Stanford2D3D \
  --output-root /data/zhuzhaoziao/RELPlus/outputs/cmx_rel_integration/rel \
  --alpha 45 --lam 0.5 --workers 1 --limit 3
```

## Real RGB/REL/label smoke layout

```bash
/data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python \
  tools/prepare_cmx_rel_smoke_data.py \
  --generation-manifest \
  /data/zhuzhaoziao/RELPlus/outputs/cmx_rel_integration/rel/generation_manifest.csv \
  --semantic-labels \
  /data/zhuzhaoziao/cmx/raw/reference_repos/2D-3D-Semantics/assets/semantic_labels.json \
  --output-root \
  /data/zhuzhaoziao/RELPlus/outputs/cmx_rel_integration/smoke_dataset \
  --limit 3
```

## Source equivalence, channel, data and model smoke test

```bash
CUDA_VISIBLE_DEVICES=0 \
/data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python \
  tests/run_rel_integration_smoke.py \
  --reference-root /home/zhuzhaoziao/RELPlus/REL-SF4PASS-reference \
  --compatibility-hha /data/bxh_copy/Pano_MA_Seg/utils/hha_util.py \
  --generation-manifest \
  /data/zhuzhaoziao/RELPlus/outputs/cmx_rel_integration/rel/generation_manifest.csv \
  --smoke-dataset-root \
  /data/zhuzhaoziao/RELPlus/outputs/cmx_rel_integration/smoke_dataset \
  --artifact-root \
  /data/zhuzhaoziao/RELPlus/outputs/cmx_rel_integration/tests \
  --limit 3 --device cuda:0
```

This command performed exactly one forward, one loss, one backward and one
optimizer step in memory. It did not save a checkpoint or enter an epoch.

## Model directory comparison

```bash
diff -qr \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  /home/zhuzhaoziao/RELPlus/CMX/models \
  /home/zhuzhaoziao/RELPlus/CMX-REL/models
```
