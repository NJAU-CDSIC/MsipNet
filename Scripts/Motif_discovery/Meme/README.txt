运行如下命令安装meme环境:
conda env create -f meme.yml

环境安装完成后，使用运行motif.py得到的结果，使用meme命令得到RBP的motif，以RBM15_HepG2为例:
meme /scripts/motif/results/split_seq/RBM15_HepG2_top_1000_mers.fasta -rna -oc /scripts/motif/results/RBM15_HepG2_8 -nmotifs 5 -minw 8 -maxw 8
