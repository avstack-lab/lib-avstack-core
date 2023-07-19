# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-04-03
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-04-03
# @Description:
"""

"""

import os
import subprocess
from shutil import copyfile

import psutil


def kill_process(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


def run_command(call_list, dirchange=None):
    curdir = os.getcwd()
    if dirchange is not None:
        os.chdir(dirchange)
    try:
        print("STARTING PROCESS")
        pro = subprocess.Popen(call_list, shell=False, preexec_fn=os.setsid)
        pro.wait()
    except KeyboardInterrupt:
        print("KILLING PROCESS")
        kill_process(pro.pid)
    else:
        print("FINISHED PROCESS")
    finally:
        os.chdir(curdir)


def copy_src_dest(src_folder, dest_folder, common_suffix):
    src_file = os.path.join(src_folder, common_suffix)
    dest_file = os.path.join(dest_folder, common_suffix)
    copyfile(src_file, dest_file)
