cp * amd/
cd amd
sbatch run.sh
sbatch run_hwc.sh
cd ..
cp * intel/
cd intel
sbatch run.sh
sbatch run_hwc.sh
cd ..