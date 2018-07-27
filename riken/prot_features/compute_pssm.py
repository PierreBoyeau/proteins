import os
import argparse
import pandas as pd
from Bio import SeqIO
import subprocess
import shlex
import dask.dataframe as dd
from dask.multiprocessing import get

from riken.protein_io.reader import get_seqrecord

COMMAND = "psiblast -db {db} \
-evalue 0.001 \
-query {query_path} \
-num_iterations 3 \
-out_ascii_pssm {query_pssm_output} \
-out {query_output}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='path to saved data')
    parser.add_argument('--db', help='data used for pssm matrix construction', default='./psiblast/swissprot/swissprot')
    parser.add_argument('--jobs', help='Number of jobs', default=16, type=int)
    parser.add_argument('--save_dir', help='path where pssm need to be saved')
    return parser.parse_args()


def save_protein_to_fasta(ptn):
    record = get_seqrecord(ptn)
    record_path = '/tmp/{}.fasta'.format(ptn.name)
    SeqIO.write(record, record_path, format='fasta')
    return record_path


def protein_routine(ptn):
    record_pt = save_protein_to_fasta(ptn)
    output_pssm = '{}_pssm.txt'.format(ptn.name)
    output = '{}_results.txt'.format(ptn.name)
    output_pssm = os.path.join(args.save_dir, output_pssm)
    output = os.path.join(args.save_dir, output)
    cmd = COMMAND.format(db=args.db, query_path=record_pt, query_pssm_output=output_pssm, query_output=output)
    cmd = shlex.split(cmd)
    subprocess.run(cmd)
    return


if __name__ == '__main__':
    args = parse_args()
    df = pd.read_csv(args.data_path, sep='\t')
    # df.apply(protein_routine, axis=1)

    ddata = dd.from_pandas(df, npartitions=args.jobs)
    res = ddata.map_partitions(lambda df: df.apply((lambda row: protein_routine(row)), axis=1))\
        .compute(get=get)



