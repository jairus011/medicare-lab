"""Execute with Jupyter, or explicit in-process IPython without kernel sockets."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]


def in_process(nb):
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    shell=InteractiveShell.instance()
    count=0
    for index,cell in enumerate(nb.cells):
        if cell.cell_type!='code':continue
        count+=1
        with capture_output(stdout=True,stderr=True,display=True) as captured:
            result=shell.run_cell(cell.source,store_history=True)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f'Notebook cell {index} failed: {result.error_before_exec or result.error_in_exec}\n{captured.stdout}\n{captured.stderr}')
        outputs=[]
        if captured.stdout:outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr:outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
        for item in captured.outputs:
            outputs.append(nbformat.v4.new_output('display_data',data=item.data,metadata=item.metadata))
        cell.outputs=outputs
        cell.execution_count=count
    nb.metadata['execution']={'method':'in-process IPython; actual cells and rich outputs','python':sys.version.split()[0],'successful_code_cells':count}
    return nb


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-process',action='store_true',help='Execute real cells through IPython in this process')
    args=parser.parse_args()
    path=ROOT/'notebooks/healthcare_final.ipynb'
    nb=nbformat.read(path,as_version=4)
    os.chdir(ROOT)
    if args.in_process:
        nb=in_process(nb)
    else:
        with tempfile.TemporaryDirectory(prefix='healthcare_kernel_') as temporary:
            kernel_path=Path(temporary)/'kernels'/'healthcare'
            kernel_path.mkdir(parents=True)
            (kernel_path/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Healthcare Python','language':'python'}))
            old=os.environ.get('JUPYTER_PATH')
            os.environ['JUPYTER_PATH']=temporary+(os.pathsep+old if old else '')
            try:
                client=NotebookClient(nb,timeout=300,kernel_name='healthcare',resources={'metadata':{'path':str(ROOT)}})
                client.execute()
                nb.metadata['execution']={'method':'Jupyter kernel via nbclient','python':sys.version.split()[0]}
            finally:
                if old is None:os.environ.pop('JUPYTER_PATH',None)
                else:os.environ['JUPYTER_PATH']=old
    nbformat.validate(nb)
    nbformat.write(nb,path)
    print(f'Executed {len(nb.cells)} cells without errors.')


if __name__=='__main__':main()
