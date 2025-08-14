VAL_FILE="massivekb_data/scaling_data_max_100000/val.mgf"
TEST_FILE="massivekb_data/scaling_data_max_100000/test.mgf"

for NUM_SPECTRA in 1 2 5 10 20; do
  for NUM_PEP in 100000 250000 500000 750000 1000000; do
    TRAIN_FILE="massivekb_data/scaling_data_max_100000/train_${NUM_SPECTRA}s_${NUM_PEP}p.mgf"
    OUTPUT_DIR="logs/casanovo_train_subsets/${NUM_SPECTRA}s_${NUM_PEP}p/"
    PBS_LOG_FILE="hpc_scripts/train_subsets/${NUM_SPECTRA}s_${NUM_PEP}p_$(date +%y%m%d%H%M%S%4N).out"
    qsub -o $PBS_LOG_FILE -j oe hpc_scripts/train_subsets/train_subset.pbs -v TRAIN_FILE=$TRAIN_FILE,VAL_FILE=$VAL_FILE,TEST_FILE=$TEST_FILE,OUTPUT_DIR=$OUTPUT_DIR
    echo -o $PBS_LOG_FILE -j oe hpc_scripts/train_subsets/train_subset.pbs -v TRAIN_FILE=$TRAIN_FILE,VAL_FILE=$VAL_FILE,TEST_FILE=$TEST_FILE,OUTPUT_DIR=$OUTPUT_DIR
  done
done