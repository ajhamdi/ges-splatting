python train_ges.py -s <YOUR_PATH>/nerf_360/bicycle -i images_4 -m ./eval/bicycle --quiet --eval --test_iterations -1 
python train_ges.py -s <YOUR_PATH>/nerf_360/flowers -i images_4 -m ./eval/flowers --quiet --eval --test_iterations -1 
python train_ges.py -s <YOUR_PATH>/nerf_360/garden -i images_4 -m ./eval/garden --quiet --eval --test_iterations -1 
python train_ges.py -s <YOUR_PATH>/nerf_360/stump -i images_4 -m ./eval/stump --quiet --eval --test_iterations -1 
python train_ges.py -s <YOUR_PATH>/nerf_360/treehill -i images_4 -m ./eval/treehill --quiet --eval --test_iterations -1 
python train_ges.py -s <YOUR_PATH>/nerf_360/room -i images_2 -m ./eval/room --quiet --eval --test_iterations -1 
python train_ges.py -s <YOUR_PATH>/nerf_360/counter -i images_2 -m ./eval/counter --quiet --eval --test_iterations -1 
python train_ges.py -s <YOUR_PATH>/nerf_360/kitchen -i images_2 -m ./eval/kitchen --quiet --eval --test_iterations -1 
python train_ges.py -s <YOUR_PATH>/nerf_360/bonsai -i images_2 -m ./eval/bonsai --quiet --eval --test_iterations -1 
python train_ges.py -s <YOUR_PATH>dev/active/lafi/lafi/tandt_db/tandt/truck -m ./eval/truck --quiet --eval --test_iterations -1 
python train_ges.py -s <YOUR_PATH>dev/active/lafi/lafi/tandt_db/tandt/train -m ./eval/train --quiet --eval --test_iterations -1 
python train_ges.py -s <YOUR_PATH>dev/active/lafi/lafi/tandt_db/db/drjohnson -m ./eval/drjohnson --quiet --eval --test_iterations -1 
python train_ges.py -s <YOUR_PATH>dev/active/lafi/lafi/tandt_db/db/playroom -m ./eval/playroom --quiet --eval --test_iterations -1 
python render.py --iteration 7000 -s <YOUR_PATH>/nerf_360/bicycle -m ./eval/bicycle --quiet --eval --skip_train
python render.py --iteration 30000 -s <YOUR_PATH>/nerf_360/bicycle -m ./eval/bicycle --quiet --eval --skip_train
python render.py --iteration 7000 -s <YOUR_PATH>/nerf_360/flowers -m ./eval/flowers --quiet --eval --skip_train
python render.py --iteration 30000 -s <YOUR_PATH>/nerf_360/flowers -m ./eval/flowers --quiet --eval --skip_train
python render.py --iteration 7000 -s <YOUR_PATH>/nerf_360/garden -m ./eval/garden --quiet --eval --skip_train
python render.py --iteration 30000 -s <YOUR_PATH>/nerf_360/garden -m ./eval/garden --quiet --eval --skip_train
python render.py --iteration 7000 -s <YOUR_PATH>/nerf_360/stump -m ./eval/stump --quiet --eval --skip_train
python render.py --iteration 30000 -s <YOUR_PATH>/nerf_360/stump -m ./eval/stump --quiet --eval --skip_train
python render.py --iteration 7000 -s <YOUR_PATH>/nerf_360/treehill -m ./eval/treehill --quiet --eval --skip_train
python render.py --iteration 30000 -s <YOUR_PATH>/nerf_360/treehill -m ./eval/treehill --quiet --eval --skip_train
python render.py --iteration 7000 -s <YOUR_PATH>/nerf_360/room -m ./eval/room --quiet --eval --skip_train
python render.py --iteration 30000 -s <YOUR_PATH>/nerf_360/room -m ./eval/room --quiet --eval --skip_train
python render.py --iteration 7000 -s <YOUR_PATH>/nerf_360/counter -m ./eval/counter --quiet --eval --skip_train
python render.py --iteration 30000 -s <YOUR_PATH>/nerf_360/counter -m ./eval/counter --quiet --eval --skip_train
python render.py --iteration 7000 -s <YOUR_PATH>/nerf_360/kitchen -m ./eval/kitchen --quiet --eval --skip_train
python render.py --iteration 30000 -s <YOUR_PATH>/nerf_360/kitchen -m ./eval/kitchen --quiet --eval --skip_train
python render.py --iteration 7000 -s <YOUR_PATH>/nerf_360/bonsai -m ./eval/bonsai --quiet --eval --skip_train
python render.py --iteration 30000 -s <YOUR_PATH>/nerf_360/bonsai -m ./eval/bonsai --quiet --eval --skip_train
python render.py --iteration 7000 -s <YOUR_PATH>dev/active/lafi/lafi/tandt_db/tandt/truck -m ./eval/truck --quiet --eval --skip_train
python render.py --iteration 30000 -s <YOUR_PATH>dev/active/lafi/lafi/tandt_db/tandt/truck -m ./eval/truck --quiet --eval --skip_train
python render.py --iteration 7000 -s <YOUR_PATH>dev/active/lafi/lafi/tandt_db/tandt/train -m ./eval/train --quiet --eval --skip_train
python render.py --iteration 30000 -s <YOUR_PATH>dev/active/lafi/lafi/tandt_db/tandt/train -m ./eval/train --quiet --eval --skip_train
python render.py --iteration 7000 -s <YOUR_PATH>dev/active/lafi/lafi/tandt_db/db/drjohnson -m ./eval/drjohnson --quiet --eval --skip_train
python render.py --iteration 30000 -s <YOUR_PATH>dev/active/lafi/lafi/tandt_db/db/drjohnson -m ./eval/drjohnson --quiet --eval --skip_train
python render.py --iteration 7000 -s <YOUR_PATH>dev/active/lafi/lafi/tandt_db/db/playroom -m ./eval/playroom --quiet --eval --skip_train
python render.py --iteration 30000 -s <YOUR_PATH>dev/active/lafi/lafi/tandt_db/db/playroom -m ./eval/playroom --quiet --eval --skip_train
python metrics.py -m "./eval/bicycle" "./eval/flowers" "./eval/garden" "./eval/stump" "./eval/treehill" "./eval/room" "./eval/counter" "./eval/kitchen" "./eval/bonsai" "./eval/truck" "./eval/train" "./eval/drjohnson" "./eval/playroom" 
