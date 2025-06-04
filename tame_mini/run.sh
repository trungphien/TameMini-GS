

# Without Blurspit & depth reinit
# python train_without_densify.py -s /home/phien23/Workspace/data/3D/mip360/room -i images_4 --eval --imp_metric indoor;
# python train_without_densify.py -s /home/phien23/Workspace/data/3D/mip360/counter -i images_4 --eval --imp_metric indoor;
# python train_without_densify.py -s /home/phien23/Workspace/data/3D/mip360/kitchen -i images_4 --eval --imp_metric indoor;
python train.py -s /home/phien23/Workspace/data/3D/mip360/bonsai -i images_4 --eval --imp_metric indoor;


# python train_without_densify.py -s /home/phien23/Workspace/data/3D/mip360/bicycle -r 300 -i images_8 --eval --imp_metric outdoor;
python train.py -s /home/phien23/Workspace/data/3D/mip360/flowers -i images_8 --eval --imp_metric outdoor;
# python train_without_densify.py -s /home/phien23/Workspace/data/3D/mip360/garden -r 400 -i images_8 --eval --imp_metric outdoor;
# python train_without_densify.py -s /home/phien23/Workspace/data/3D/mip360/stump -r 300 -i images_8 --eval --imp_metric outdoor;
# python train_without_densify.py -s /home/phien23/Workspace/data/3D/mip360/treehill -r 400 -i images_8 --eval --imp_metric outdoor;



# python train_without_densify.py -s /home/phien23/Workspace/data/3D/TanksAndTemples/TanksAndTemples_colmap/Train -r 600 --eval --imp_metric outdoor;
# python train_without_densify.py -s /home/phien23/Workspace/data/3D/TanksAndTemples/TanksAndTemples_colmap/Truck -r 600 --eval --imp_metric outdoor;


# python train_without_densify.py -s /home/phien23/Workspace/data/3D/DeepBlending_colmap/DrJohnson/colmap -r 500 --eval --imp_metric outdoor; 
# python train_without_densify.py -s /home/phien23/Workspace/data/3D/DeepBlending_colmap/Playroom/colmap -r 700 --eval --imp_metric outdoor; 

# Without Simpify
# python train_without_simplify.py -s /home/phien23/Workspace/data/3D/mip360/room -i images_4 --eval --imp_metric indoor;
# python train_without_simplify.py -s /home/phien23/Workspace/data/3D/mip360/counter -i images_4 --eval --imp_metric indoor;
# python train_without_simplify.py -s /home/phien23/Workspace/data/3D/mip360/kitchen -i images_4 --eval --imp_metric indoor;
# python train_without_simplify.py -s /home/phien23/Workspace/data/3D/mip360/bonsai -i images_4 --eval --imp_metric indoor;


# python train_without_simplify.py -s /home/phien23/Workspace/data/3D/mip360/bicycle -r 300 -i images_8 --eval --imp_metric outdoor;
# python train_without_simplify.py -s /home/phien23/Workspace/data/3D/mip360/flowers -i images_8 --eval --imp_metric outdoor;
# python train_without_simplify.py -s /home/phien23/Workspace/data/3D/mip360/garden -r 400 -i images_8 --eval --imp_metric outdoor;
# python train_without_simplify.py -s /home/phien23/Workspace/data/3D/mip360/stump -r 300 -i images_8 --eval --imp_metric outdoor;
# python train_without_simplify.py -s /home/phien23/Workspace/data/3D/mip360/treehill -r 400 -i images_8 --eval --imp_metric outdoor;



# python train_without_simplify.py -s /home/phien23/Workspace/data/3D/TanksAndTemples/TanksAndTemples_colmap/Train -r 600 --eval --imp_metric outdoor;
# python train_without_simplify.py -s /home/phien23/Workspace/data/3D/TanksAndTemples/TanksAndTemples_colmap/Truck -r 600 --eval --imp_metric outdoor;


# python train_without_simplify.py -s /home/phien23/Workspace/data/3D/DeepBlending_colmap/DrJohnson/colmap -r 500 --eval --imp_metric outdoor; 
# python train_without_simplify.py -s /home/phien23/Workspace/data/3D/DeepBlending_colmap/Playroom/colmap -r 700 --eval --imp_metric outdoor; 

# Without coarsetofine
# python train_without_coarsetofine.py -s /home/phien23/Workspace/data/3D/mip360/room -i images_4 --eval --imp_metric indoor;
# python train_without_coarsetofine.py -s /home/phien23/Workspace/data/3D/mip360/counter -i images_4 --eval --imp_metric indoor;
# python train_without_coarsetofine.py -s /home/phien23/Workspace/data/3D/mip360/kitchen -i images_4 --eval --imp_metric indoor;
# python train_without_coarsetofine.py -s /home/phien23/Workspace/data/3D/mip360/bonsai -i images_4 --eval --imp_metric indoor;


# python train_without_coarsetofine.py -s /home/phien23/Workspace/data/3D/mip360/bicycle -r 300 -i images_8 --eval --imp_metric outdoor;
# python train_without_coarsetofine.py -s /home/phien23/Workspace/data/3D/mip360/flowers -i images_8 --eval --imp_metric outdoor;
# python train_without_coarsetofine.py -s /home/phien23/Workspace/data/3D/mip360/garden -r 400 -i images_8 --eval --imp_metric outdoor;
# python train_without_coarsetofine.py -s /home/phien23/Workspace/data/3D/mip360/stump -r 300 -i images_8 --eval --imp_metric outdoor;
# python train_without_coarsetofine.py -s /home/phien23/Workspace/data/3D/mip360/treehill -r 400 -i images_8 --eval --imp_metric outdoor;



# python train_without_coarsetofine.py -s /home/phien23/Workspace/data/3D/TanksAndTemples/TanksAndTemples_colmap/Train -r 600 --eval --imp_metric outdoor;
# python train_without_coarsetofine.py -s /home/phien23/Workspace/data/3D/TanksAndTemples/TanksAndTemples_colmap/Truck -r 600 --eval --imp_metric outdoor;


# python train_without_coarsetofine.py -s /home/phien23/Workspace/data/3D/DeepBlending_colmap/DrJohnson/colmap -r 500 --eval --imp_metric outdoor; 
# python train_without_coarsetofine.py -s /home/phien23/Workspace/data/3D/DeepBlending_colmap/Playroom/colmap -r 700 --eval --imp_metric outdoor; 