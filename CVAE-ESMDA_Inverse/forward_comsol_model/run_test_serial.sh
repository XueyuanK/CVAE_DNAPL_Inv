#BSUB -q mpi
#BSUB -n 24
#BSUB –o %J.out
#BSUB –e %J.err
comsol server -np 8 -silent -port 61036 -tmpdir temporary_files1 &
sleep 30s
matlab -nodisplay -r forward_model