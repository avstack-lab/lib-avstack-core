
set -e

JET_PATH="/usr/lib/python3.8/dist-packages"
VENV_PATH="/home/spencer/.cache/pypoetry/virtualenvs/lib-avstack-core-CUmhTGWQ-py3.8"
SITEPACK="${VENV_PATH}/lib/python3.8/site-packages"

ln -sf "${JET_PATH}/jetson_utils" "$SITEPACK"
ln -sf "${JET_PATH}/jetson_inference" "$SITEPACK"
ln -sf "${JET_PATH}/jetson_utils_python.so" "$SITEPACK"
ln -sf "${JET_PATH}/jetson_inference_python.so" "$SITEPACK"
ln -sf "${JET_PATH}/jetson" "$SITEPACK"
ln -sf "${JET_PATH}/Jetson" "$SITEPACK"