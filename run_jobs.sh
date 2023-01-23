
run_on_file(){
    task_id=$1
    file_path=../re21.9/local_run/subTask-${task_id}/myfile_tree.root
    python3 root_to_h5.py --input_file $file_path\
                        --output data/ttbar/train-${task_id}.h5
}

for i in {0..25}
do
    run_on_file $i
    echo "$i -- OK"
done
 



