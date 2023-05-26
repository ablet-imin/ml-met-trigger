#!/bin/bash -e

run_on_file_CELL(){
    python3 root_to_h5.py --input_file $1 --output $2
}

run_on_file_TCL(){
    python3 root_to_h5_TCL_vlen.py --input_file $1\
            --output $2\
             --tcl_name $3
}

run_on_file_JETS(){
    python3 root_to_h5_Jets.py --input_file $1\
            --output $2\
             --jet_name $3
}

files_path=$1
file_pattern=$2
output_dir=$3
files=$(ls $files_path)

output_path_TCL="${output_dir}/Calo422TopoClusters"
mkdir -p $output_path_TCL

output_path_TCL_SK="${output_dir}/Calo422SKTopoClusters"
mkdir -p $output_path_TCL_SK

output_path_CELL="${output_dir}/cells/"
mkdir -p $output_path_CELL

output_path_JET="${output_dir}/jets/"
mkdir -p $output_path_JET


COUNTER=0
for f in $files
do
    abs_file_name=$files_path'/'$f
    #echo $abs_file_name
    sed_tring="s/.*\/${file_pattern}/\1/g"
    file_id=$(echo "$abs_file_name" | sed "$sed_tring")

    let COUNTER++
    #last five files will be test files
    if [ "$COUNTER" -gt "20" ]; then
        #run_on_file_TCL $abs_file_name "${output_path_TCL}/test-${file_id}.h5" "Calo422TopoClusters"
        #run_on_file_TCL $abs_file_name "${output_path_TCL_SK}/test-${file_id}.h5" "Calo422SKTopoClusters"
        #cell
        #run_on_file_CELL $abs_file_name "${output_path_CELL}/test-${file_id}.h5"
        #Jets
        run_on_file_JETS $abs_file_name "${output_path_JET}/test-${file_id}.h5" "AntiKt4emtopoCalo422Jets"
    else
        #run_on_file_TCL $abs_file_name "${output_path_TCL}/train-${file_id}.h5" "Calo422TopoClusters"
        #run_on_file_TCL $abs_file_name "${output_path_TCL_SK}/train-${file_id}.h5" "Calo422SKTopoClusters"
        #cell
        #run_on_file_CELL $abs_file_name "${output_path_CELL}/train-${file_id}.h5"
        #Jets
        run_on_file_JETS $abs_file_name "${output_path_JET}/train-${file_id}.h5" "AntiKt4emtopoCalo422Jets"
    fi

    echo "$file_id ---- OK"

done



