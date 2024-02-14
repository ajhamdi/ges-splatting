
sequence="train"
path_to_dataset="tandt_db/tandt/$sequence"
path_to_outputs="output/$sequence"

densification_interval=100
iterations=30000
rotation_lr=0.001
percent_dense=0.01
densify_grad_threshold=0.0002
prune_opacity_threshold=0.005
opacity_reset_interval=3000
scaling_lr=0.005

exp_set="00"
shape_lr=0.001
shape_reset_interval=300000
shape_pruning_interval=100
prune_shape_threshold=0.01
shape_strngth=0.1

python train_laplace.py -s $path_to_dataset -m $path_to_outputs --eval --iterations $iterations --densification_interval $densification_interval --rotation_lr $rotation_lr --percent_dense $percent_dense --densify_grad_threshold $densify_grad_threshold --prune_opacity_threshold $prune_opacity_threshold --opacity_reset_interval $opacity_reset_interval --shape_lr $shape_lr --shape_reset_interval $shape_reset_interval --shape_pruning_interval $shape_pruning_interval --prune_shape_threshold $prune_shape_threshold --exp_set $exp_set --scaling_lr $scaling_lr --shape_strngth $shape_strngth

