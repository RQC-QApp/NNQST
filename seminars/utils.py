# File name: utils.py
# Original file: https://github.com/QISKit/qiskit-tutorial/blob/master/1_introduction/getting_started.ipynb
# Edited by: Anton Karazeev <a.karazeev@rqc.ru>

import os
import shutil
from qiskit.tools.visualization import latex_drawer
import pdf2image


def circuitImage(circuit, basis="u1,u2,u3,cx"):
    """Obtain the circuit in image format
    Note: Requires pdflatex installed (to compile Latex)
    Note: Required pdf2image Python package (to display pdf as image)
    """
    filename='circuit'
    tmpdir='tmp/'
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    latex_drawer(circuit, tmpdir+filename+".tex", basis=basis)
    os.system("pdflatex -output-directory {} {}".format(tmpdir, filename+".tex"))
    images = pdf2image.convert_from_path(tmpdir+filename+".pdf")
    shutil.rmtree(tmpdir)
    return images[0]


basis="u1,u2,u3,cx,x,y,z,h,s,t,rx,ry,rz"
