export CARLA_ROOT=/home/raykr/projects/ScenGE-AAAI/CARLA_0.9.13_safebench
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/agents
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
if [ -f "${CONDA_PREFIX}/lib/libtiff.so.6" ] && [ ! -e "${CONDA_PREFIX}/lib/libtiff.so.5" ]; then
  ln -sf libtiff.so.6 "${CONDA_PREFIX}/lib/libtiff.so.5"
fi
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}
