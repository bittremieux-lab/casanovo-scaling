import os

import pandas as pd
from pyteomics import mgf
from tqdm import tqdm


def add_charge(mgf_file, metadata_file, chunk_size=1000000, chunk_dir=""):
    metadata = pd.read_csv(metadata_file, sep="\t")
    metadata['mz_file'] = metadata['filename'].str.split('/').str[-1]
    metadata.set_index(['mz_file', 'scan'], inplace=True)
    metadata = metadata.sort_index()

    chunk_index = 0
    os.makedirs(chunk_dir, exist_ok=True)

    chunk = []
    with mgf.read(
            mgf_file,
            use_index=False,
            convert_arrays=0,
            read_charges=False,
            read_ions=False,
    ) as massivekb:
        for spectrum in tqdm(massivekb, total=66e6):
            if "charge" not in spectrum['params']:
                mzml_file, _, scan = spectrum['params']['title'].split(':')
                metadata_rows = metadata.loc[(mzml_file, int(scan))]
                if len(metadata_rows) == 0:
                    print(f"Did not find metadata for {mzml_file}")
                elif len(metadata_rows) > 1:
                    print(f"Found multiple metadata for {mzml_file}")
                    print(metadata_rows['filename'])
                else:
                    charge = metadata_rows.iloc[0]['charge']
                    spectrum['params']['charge'] = charge
            chunk.append(spectrum)
            if len(chunk) >= chunk_size:
                out_file = os.path.join(chunk_dir, f"chunk_{chunk_index}.mgf")
                mgf.write(chunk, out_file, use_numpy=True)
                chunk_index += 1
                chunk = []
        if chunk:
            out_file = os.path.join(chunk_dir, f"chunk_{chunk_index}.mgf")
            mgf.write(chunk, out_file, use_numpy=True)


if __name__ == "__main__":
    mgf_file = "/mnt/data/cdens/casanovo-scaling/massivekb_data/massiveKB_3cac03860ff7453a821332ab4cff20f4.mgf"
    metadata_file = "/mnt/data/cdens/casanovo-scaling/massivekb_data/LIBRARY_AUGMENT-3cac0386-candidate_library_spectra-main.tsv"
    chunk_dir = "/mnt/data/cdens/casanovo-scaling/massivekb_data/chunks/"
    add_charge(mgf_file, metadata_file, chunk_dir=chunk_dir)
