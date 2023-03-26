import os

### Top-most directory path
home_dir  = ""


### 3D-FRONT dataset (optional)
tdfront_dir =  os.path.join(home_dir, "3D-FRONT_dataset_65347")
future_model_dir = os.path.join(tdfront_dir, "3D-FUTURE-model")
front_dir = os.path.join(tdfront_dir, "3D-FRONT")
path_to_floor_plan_textures = os.path.join(home_dir, "ATISS/demo/floor_plan_texture_images")


### Preprocessed 3D-FRONT dataset (stores files like splits.csv, each scene's boxes.npz)
data_dir = os.path.join(home_dir, "3DFRONT_65347")


### Evaluation files/output directory
eval_dir = os.path.join(home_dir, "LEGO-Net", "eval")