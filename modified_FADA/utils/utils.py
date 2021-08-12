import os
import subprocess

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    return path

def exe_cmd(cmd, verbose=True):
    """
    :return: (stdout, stderr=None)
    """
    r = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ret = r.communicate()
    r.stdout.close()
    if verbose:
        res = str(ret[0].decode()).strip()
        if res:
            print(res)
    if ret[1] is not None:
        print(str(ret[0].decode()).strip())
    return ret

def get_domains(task):
    domains = ["./domain_adaptation_images/webcam/images/",
               "./domain_adaptation_images/dslr/images/",
               "./domain_adaptation_images/amazon/images/"]
    query = task.split('2')
    ret = []
    for q in query:
        if q == 'w':
            ret.append(domains[0])
        elif q == 'd':
            ret.append(domains[1])
        elif q == 'a':
            ret.append(domains[2])
    return ret[0], ret[1]