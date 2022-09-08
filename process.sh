domains=(humans books songs)
for domain in ${domains[*]}; do
    new_path=./data_release/${domain}/processed_data_100
    if [ -d "${new_path}" ]; then rm -rf ${new_path}; fi
    rm -rf ./data_release/${domain}/processed_data
    python preprocess.py ./data_release ${domain}
    mv ./data_release/${domain}/processed_data ${new_path}
    done
