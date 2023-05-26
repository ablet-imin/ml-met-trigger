
run_on_file(){
    python3 root_to_h5.py --input_file $1 --output $2
}
 
 
files_path=$1
file_pattern="$2"
output_dir=$3
files=$(ls $files_path)

cell_name="cells"


output_path="${output_dir}/${cell_name}/"
mkdir -p $output_path


for f in $files
do
    abs_file_name=$files_path'/'$f
    echo $abs_file_name
    sed_tring="s/.*\/${file_pattern}/\1/g"
    file_id=$(echo "$abs_file_name" | sed "$sed_tring")
    
    #last five files will be test files
    if [ "$file_id" -gt "20" ]; then
        run_on_file $abs_file_name "${output_path}/test-${file_id}.p5"
    else
        run_on_file $abs_file_name "${output_path}/train-${file_id}.p5"
    fi
    
    echo "$file_id ---- OK"
    
done



