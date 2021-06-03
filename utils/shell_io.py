from subprocess import run, PIPE, Popen
from os.path import join

def byte_style(content, fg_color = '7', bg_color = '0', bold = False, dim = False, negative = False, underlined = False, blink = False):
    prefix = '\033['
    if not (bold or dim or negative or underlined or blink):
        prefix += '0;'
    if bold:
        prefix += '1;'
    if dim:
        prefix += '2;'
    if negative:
        prefix += '3;'
    if underlined:
        prefix += '4;'
    if blink:
        prefix += '5;'

    prefix += f'3{fg_color};4{bg_color}m'
    return prefix + content + '\033[m'

def call_fasttext(fasttext, wfile, vfile, ft_bin, ft_lower): # TODO: async
    # import pdb; pdb.set_trace()
    src = Popen(['cat', wfile], stdout = PIPE)
    src = Popen(['cut', '-f1'], stdin = src.stdout, stdout = PIPE)
    if ft_lower:
        src = Popen(['tr', '[:upper:]', '[:lower:]'], stdin = src.stdout, stdout = PIPE)
    dst = Popen([fasttext, 'print-word-vectors', ft_bin], stdin = src.stdout, stdout = PIPE)
    with open(vfile, 'wb') as fw: # TODO: save it in numpy format!
        dst = Popen(['cut', '-d ', '-f2-'], stdin = dst.stdout, stdout = fw) # space is intersting

def parseval(cmd_tuple, fhead, fdata):
    command = list(cmd_tuple)
    command.append(fhead)
    command.append(fdata)
    return run(command, stdout = PIPE, stderr = PIPE)

from collections import defaultdict
def rpt_summary(rpt_lines, get_individual, get_summary):
    individuals = get_individual
    summary = defaultdict(int)
    for line in rpt_lines.split('\n'):
        if line.startswith('===='):
            if individuals is True: # start
                individuals = []
            elif isinstance(individuals, list): # end
                individuals = tuple(individuals)
                if not get_summary:
                    return individuals
        elif isinstance(individuals, list):
            sent = tuple(float(s) if '.' in s else int(s) for s in line.split())
            individuals.append(sent)
        elif get_summary:
            if line.startswith('Number of sentence'):
                summary['N'] = int(line[line.rfind(' '):])
            if line.startswith('Bracketing Recall'):
                summary['LR'] = float(line[line.rfind(' '):])
            if line.startswith('Bracketing Precision'):
                summary['LP'] = float(line[line.rfind(' '):])
            if line.startswith('Bracketing FMeasure'):
                summary['F1'] = float(line[line.rfind(' '):])
            if line.startswith('Tagging accuracy'):
                summary['TA'] = float(line[line.rfind(' '):])
                break
            # ID  Len.  Stat. Recal  Prec.  Bracket gold test Bracket Words  Tags Accrac
    if get_individual:
        return individuals, summary
    return summary

def concatenate(src_files, dst_file):
    command = ['cat']
    command.extend(src_files)
    rs = run(command, stdout = PIPE, stderr = PIPE)
    with open(dst_file, 'wb') as fw:
        fw.write(rs.stdout)
    assert not rs.stderr

def has_discodop():
    try:
        command = ['discodop', 'eval']
        dst = run(command, stdout = PIPE, stderr = PIPE)
    except:
        return False
    return True
    
def discodop_eval(fhead, fdata, prm_file, rpt_file = None):
    command = ['discodop', 'eval', fhead, fdata, prm_file]
    dst = run(command, stdout = PIPE, stderr = PIPE)
    total = dst.stdout.decode('ascii')
    smy_string = total.rfind(' Summary ')
    smy_string = total[smy_string:].split('\n')
    smy = dict(TF = 0, TP = 0, TR = 0, DF = 0, DP = 0, DR = 0)
    for line in smy_string:
        if line.startswith('labeled recall:'):
            smy['TR'] = float(line.split()[2])
        elif line.startswith('labeled precision:'):
            smy['TP'] = float(line.split()[2])
        elif line.startswith('labeled f-measure:'):
            smy['TF'] = float(line.split()[2])
    command.append('--disconly')
    dst = run(command, stdout = PIPE, stderr = PIPE)
    discontinuous = dst.stdout.decode('ascii')
    smy_string = discontinuous.rfind(' Summary ')
    discontinuous = discontinuous[smy_string:]
    for line in discontinuous.split('\n'):
        if line.startswith('labeled recall:'):
            smy['DR'] = float(line.split()[2])
        elif line.startswith('labeled precision:'):
            smy['DP'] = float(line.split()[2])
        elif line.startswith('labeled f-measure:'):
            smy['DF'] = float(line.split()[2])

    if rpt_file:
        rpt_file.write('\n═══════════════════════════════════════════════════════════════════════════════════════\n')
        rpt_file.write('Results from discodop eval: [Total]\n')
        rpt_file.write(total)
        rpt_file.write('\n [Discontinuous Only]\n')
        rpt_file.write(discontinuous)
    return smy