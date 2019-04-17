# mlx

Machine learning experiments using PyTorch and fastai

### Build and run Docker image
```
./scripts/build
./scripts/console
```

### Run Jupyter locally
```
jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
```

### Feature Visualization

A [notebook](mlx/feat_viz/min_feat_viz.ipynb) with a minimal example of doing feature visualization.

### Camvid-Tiramisu Semantic Segmentation

This is a minimal example of a script for training a model using fastai/PyTorch with options to run a small test locally and then run it on a GPU using AWS Batch with data and results synced to S3. If the job is stopped in the middle of training, re-running it will resume training from a saved checkpoint. If you would like to force it to train from scratch you will need to delete the saved training results first. This takes about a minute to get to around 90% accuracy on a p3.2xlarge.

#### Prep
* Build dataset zip file by cloning https://github.com/alexgkendall/SegNet-Tutorial.git and making a zip file containing just the `CamVid` subdirectory. Upload zip file to S3.
* Setup Batch resources (job def, job queue, compute environment, ECR repo) using CloudFormation template in [raster-vision-aws](https://github.com/azavea/raster-vision-aws). This was intended for use with [Raster Vision](https://github.com/azavea/raster-vision), but can be used to setup GPU resources more generally.
* Adjust constants in `./scripts/publish_image` and `./scripts/job_def.json`
* Run `./scripts/add_job_def`. This is needed to add a special job def which maps `/dev/shm` which is needed to use multiprocessing and PyTorch on Batch.
* Adjust URIs and hyperparams in `mlx.semseg.camvid`.

#### Run test locally
```
./script/build
./scripts/console
python -m mlx.semseg.camvid --test
```

#### Run on Batch
```
./scripts/build
./scripts/publish-image
./scripts/console
python -m mlx.semseg.camvid --batch
```

### Object Detection

Work in progress on writing a single shot object detector.

* Single box regression using Pascal 2007 [notebook](mlx/od/nbs/pascal_regression.ipynb)
* Various utility functions for dealing with boxes can be tested using
 `python -m mlx.od.test_utils`