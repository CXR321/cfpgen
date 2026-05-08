import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from ast import literal_eval
import subprocess

start_idx_dict = {
    '1prw': [15, 51],
    '1bcf': [17, 46, 90, 122],
    '5tpn': [108],
    '3ixt': [0],
    '4jhw': [37, 144],
    '4zyp': [357],
    '5wn9': [1],
    '5ius': [34, 88],
    '5yui': [89, 114, 194],
    '6vw1': [5, 45],
    '1qjg': [13, 37, 98],
    '1ycr': [2],
    '2kl8': [0, 27],
    '7mrx': [25],
    '5trv': [45],
    '6e6r': [22],
    '6exz': [25],
}

end_idx_dict = {
    '1prw': [34, 70],
    '1bcf': [24, 53, 98, 129],
    '5tpn': [126],
    '3ixt': [23],
    '4jhw': [43, 159],
    '4zyp': [371],
    '5wn9': [20],
    '5ius': [53, 109],
    '5yui': [93, 116, 196],
    '6vw1': [23, 63],
    '1qjg': [13, 37, 98],
    '1ycr': [10],
    '2kl8': [6, 78],
    '7mrx': [46],
    '5trv': [69],
    '6e6r': [34],
    '6exz': [39],
}


def cal_success_scaffold(pdb):
    total = len(pdb)
    pdb['total'] = total
    pdb = pdb[(pdb['rmsd'] < 1.0) & (pdb['plddt'] > 70)]
    return pdb


def calc_rmsd_tmscore(pdb_name, reference_PDB, scaffold_pdb_path=None, scaffold_info_path=None, ref_motif_starts=[30], ref_motif_ends=[44],
                      output_path=None):
    "Calculate RMSD between reference structure and generated structure over the defined motif regions"

    motif_df = pd.read_csv(os.path.join(scaffold_info_path, f'{pdb_name}.csv'), index_col=0) #, nrows=num_structures)
    results = []
    for pdb in sorted(os.listdir(os.path.join(scaffold_pdb_path, f'{pdb_name}'))): # This needs to be in numerical order to match new_starts file
        if not pdb.endswith('.pdb'):
            continue
        ref = mda.Universe(reference_PDB)
        predict_PDB = os.path.join(os.path.join(scaffold_pdb_path, f'{pdb_name}'), pdb)
        u = mda.Universe(predict_PDB)

        ref_selection = f'name CA and resnum ' #f'name CA and segid {chain_id} and resnum '
        u_selection = f'name CA and resnum '
        i = int(pdb.split('_')[1])
        new_motif_starts = literal_eval(motif_df['start_idxs'].iloc[i])
        new_motif_ends = literal_eval(motif_df['end_idxs'].iloc[i])

        for j in range(len(ref_motif_starts)):
            ref_selection += str(ref_motif_starts[j]) + ':' + str(ref_motif_ends[j]) + ' '
            u_selection += str(new_motif_starts[j]+1) + ':' + str(new_motif_ends[j]+1) + ' '
        # print("U SELECTION", u_selection)
        # print("SEQUENCE", i)
        # print("ref", ref.select_atoms(ref_selection).resnames)
        # print("gen", u.select_atoms(u_selection).resnames)
        # This asserts that the motif sequences are the same - if you get this error something about your indices are incorrect - check chain/numbering
        assert len(ref.select_atoms(ref_selection).resnames) == len(u.select_atoms(u_selection).resnames), "Motif lengths do not match, check PDB preprocessing for extra residues"

        if (ref.select_atoms(ref_selection).resnames == u.select_atoms(u_selection).resnames).all():
            rmsd = rms.rmsd(u.select_atoms(u_selection).positions,
                            ref.select_atoms(ref_selection).positions,
                            center=True,  # subtract the center of geometry
                            superposition=True)  # superimpose coordinates
        else:
            backbone_ref = ref.select_atoms(ref_selection).select_atoms('name N or name CA or name C')
            backbone_u = u.select_atoms(u_selection).select_atoms('name N or name CA or name C')
            rmsd = rms.rmsd(backbone_u.positions,
                            backbone_ref.positions,
                            center=True,  # subtract the center of geometry
                            superposition=True)  # superimpose coordinates


    # 计算 TM-score
        temp_file = open(os.path.join(output_path, 'temp_tmscores.txt'), 'w')
        subprocess.call(['./TMscore', reference_PDB, predict_PDB,  '-seq'], stdout=temp_file)
        with open(os.path.join(output_path, 'temp_tmscores.txt'), 'r') as f:
            for line in f:
                if len(line.split()) > 1 and "TM-score" == line.split()[0]:
                    tm_score = line.split()[2]
                    break

        plddt = float(predict_PDB.split('_')[-1][:-4])
        results.append((pdb_name, i, rmsd, plddt, tm_score))

    return results



if __name__ == '__main__':
    # scaffold_dir = "generation-results/dplm_650m_scaffold"
    scaffold_dir = "/home/yinj0b/repository/dplm/generation-results/dplm-650m_go-87_ipr241_seq-controlnet-v1-mask95_scaffold_num20_controlnet-fix/"
    motif_pdb_dir = '/home/yinj0b/repository/dplm/data-bin/scaffolding-pdbs'
    output_dir = os.path.join(scaffold_dir, 'eval_results')
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for pdb in start_idx_dict.keys():
        # if not pdb == '4jhw':
        #     continue
        print(pdb)
        ref_motif_starts = start_idx_dict[pdb]
        ref_motif_ends = end_idx_dict[pdb]
        reference_PDB = os.path.join(motif_pdb_dir, pdb+'_reference.pdb')
        if not os.path.isfile(reference_PDB):
            continue
        with open(reference_PDB) as f:
            line = f.readline()
            ref_basenum = int(line.split()[5])
        ref_motif_starts = [num + ref_basenum for num in ref_motif_starts]
        ref_motif_ends = [num + ref_basenum for num in ref_motif_ends]
        try:
            result = calc_rmsd_tmscore(
                pdb_name=pdb,
                reference_PDB=reference_PDB,
                scaffold_pdb_path=f'{scaffold_dir}/scaffold_fasta/esmfold_pdb',
                scaffold_info_path=f'{scaffold_dir}/scaffold_info',
                ref_motif_starts=ref_motif_starts,
                ref_motif_ends=ref_motif_ends,
                output_path=output_dir,
            )
            results += result
        except AssertionError as e:
            print(f"AssertionError encountered for {pdb}: {e}")
            continue

    results = pd.DataFrame(results, columns=['pdb_name', 'index', 'rmsd', 'plddt', 'tmscore'])
    results.to_csv(os.path.join(output_dir, 'rmsd_tmscore.csv'), index=False)
    rmsd_tmscore = pd.read_csv(os.path.join(output_dir, 'rmsd_tmscore.csv'))

    success_scaffold = rmsd_tmscore.groupby('pdb_name', as_index=False).apply(cal_success_scaffold)
    # success_scaffold.to_csv(os.path.join(output_dir, 'success_scaffold.csv'), index=False)
    success_scaffold_count = success_scaffold.groupby('pdb_name').size()
    success_scaffold_count = success_scaffold_count.reset_index(name='success_count')

    all_pdb = list(rmsd_tmscore['pdb_name'].unique())
    success_pdb = list(success_scaffold_count['pdb_name'])
    failed_pdb = list(set(all_pdb)-set(success_pdb))
    failed_scaffold_count = {
        'pdb_name': failed_pdb,
        'success_count': [0] * len(failed_pdb),
    }
    results = pd.concat([success_scaffold_count, pd.DataFrame(failed_scaffold_count)]).sort_values('pdb_name')
    print(results)
    results.to_csv(os.path.join(output_dir, 'result.csv'))