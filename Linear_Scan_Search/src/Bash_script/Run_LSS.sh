# Arwa Wali (amw7@njit.edu)

# dataset path
p_datasets="/ALOI-100-Dataset/"
pwd=${p_datasets}
output_dir=${p_datasets}"_IH_App_"

echo $output_dir
# dataset name
dataset="ALOI-100"
K_dash=200
#cluster_max_size=1000
#outlier_threshold=5
#Iter=61
#z=0.05
#spa=$((/afs/cad/linux/anaconda-2.1.0/anaconda/bin/python /phenome/amw7/Sparsification-Kmeans/src/Dataset_Editing/calculate_z.py  ${pwd} ${dataset} ${z}) 2>&1)
#echo ${spa} 
COUNTER=0
while [  $COUNTER -lt 1 ]; do
        echo The counter is $COUNTER
        output=$(java -jar -Xms256m -Xmx5g -XX:-UseGCOverheadLimit "/LID-Fingerprint/Linear_Scan_Search/dist/Linear_Scan_Search.jar" ${p_datasets} ${dataset} ${K_dash})
        # save output to file
        destdir=${output_dir}${COUNTER}
        echo "$output" > "$destdir"
        let COUNTER=COUNTER+1 
done

